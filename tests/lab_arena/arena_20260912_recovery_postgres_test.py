"""Exact recovery of the cancelled 2026-09-12 Arena execution stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import scoring
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    complete,
    encrypted_runtime_credentials,
    hotkey,
    pass_code_review,
    round_config,
    sha,
    source_submission_doc,
)


ROUND_ID = "arena-2026-09-12"
FAILED_SUBMISSION = "sub-d39d12f74f5f6fb56837d22890c7dd94"
PRESERVED_SUBMISSION = "sub-ddd33c5c142f317e0279265f179b5bc1"
RUNNER = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
CREDENTIAL_FAILURE_AT = "2026-09-12T00:22:34.933602Z"
PARTICIPANT_IDS = (
    "sub-67fdb5d43bc8f9ea4cb6e74df4797037",
    "sub-c2ab2d55f187807c2793433701485825",
    "sub-1460c4e33f19f923b969ef2fd941072f",
    FAILED_SUBMISSION,
    "sub-eb8a764bc501efb7d47f9c6c74d49756",
    "sub-64750217950f089a3bec21ee6fe72d6e",
    PRESERVED_SUBMISSION,
    "baseline-2026-09-12",
)
REPAIR_RUN_IDS = tuple(
    f"{ROUND_ID}:{FAILED_SUBMISSION}:1:{position}:{attempt}"
    for position, attempts in ((4, (2,)), (5, (2,)), (6, (1, 2)),
                               (7, (1, 2)), (8, (1, 2)), (9, (1, 2)))
    for attempt in attempts
)
MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "219-recover-arena-2026-09-12-credential-failure.sql"
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _connect(database):
    psycopg2, dsn = database
    return psycopg2.connect(**dsn)


def _reset(database) -> None:
    connection = _connect(database)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "TRUNCATE public.lab_arena_ledger, public.lab_arena_rounds "
                "RESTART IDENTITY CASCADE"
            )
    finally:
        connection.close()


def _row_hash(cursor, table: str, where: str, params=()) -> str:
    cursor.execute(
        f"SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg("  # noqa: S608
        f"pg_catalog.to_jsonb(row_value)::text, '|' "
        f"ORDER BY pg_catalog.to_jsonb(row_value)::text), '')) "
        f"FROM (SELECT * FROM public.{table} WHERE {where}) AS row_value",
        params,
    )
    return cursor.fetchone()[0]


def _apply(connection, *, error: str | None = None) -> None:
    with connection.cursor() as cursor:
        if error is None:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            return
        with pytest.raises(Exception, match=error):
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        cursor.execute("ROLLBACK")


def _seed(database, *, wrong_refusal: bool = False, live_call: bool = False,
          publication_started: bool = False):
    _reset(database)
    connection = _connect(database)
    connection.autocommit = True
    transport = PsycopgTransport(lambda: _connect(database))
    store = ArenaStore(transport, lease_ttl_seconds=120)
    miner_hotkeys = {
        submission_id: hotkey(f"219-{index}")
        for index, submission_id in enumerate(PARTICIPANT_IDS)
    }
    config = round_config(
        ROUND_ID,
        [RUNNER],
        stage_1_icps=10,
        max_attempts=2,
        execution_cap_microusd=50_000_000,
        scoring_cap_microusd=50_000_000,
        rewards_enabled=True,
        cost_per_company_microusd=500_000,
    )
    stored_schedule = {
        "submission_open": "2026-09-11T00:00:00Z",
        "submission_cutoff": "2026-09-12T00:00:00Z",
        "benchmark_deadline": "2026-09-12T00:30:00Z",
        "stage_1_start": "2026-09-12T00:30:01Z",
        "stage_1_close": "2026-09-12T04:30:01Z",
        "stage_1_scoring_close": "2026-09-12T11:00:01Z",
        "stage_2_start": "2026-09-12T11:00:02Z",
        "stage_2_close": "2026-09-12T14:00:02Z",
        "final_scoring_close": "2026-09-12T20:30:02Z",
        "publication_deadline": "2026-09-12T20:30:03Z",
    }
    config.update({
        "max_challengers": 8,
        "companies_per_icp": 5,
        "runner_slot_ceiling": 8,
        "icp_wall_clock_seconds": 300,
        "scoring_wall_clock_seconds": 900,
    })
    assert store.create_round(ROUND_ID, config)["status"] == "created"
    participants = []
    for submission_id in PARTICIPANT_IDS:
        miner = miner_hotkeys[submission_id]
        is_baseline = submission_id == "baseline-2026-09-12"
        assert store.register_submission(
            ROUND_ID,
            submission_id,
            miner,
            source_submission_doc(ROUND_ID, submission_id, is_king=is_baseline),
        )["status"] == "registered"
        assert store.accept_submission_with_credentials(
            ROUND_ID,
            submission_id,
            miner,
            encrypted_runtime_credentials(submission_id),
        )["status"] == "ok"
        pass_code_review(store, submission_id, miner)
        assert store.update_submission(
            ROUND_ID, submission_id, "accepted", "frozen",
            {"is_king": is_baseline},
        )["status"] == "ok"
        participants.append({
            "submission_id": submission_id,
            "miner_hotkey": miner,
            "is_king": is_baseline,
        })
    assert store.transition_round(
        ROUND_ID,
        "open",
        "committed",
        {
            "participants": participants,
            "benchmark_ref": f"arena/{ROUND_ID}/benchmark.json",
            "evaluation_date": "2026-09-12",
        },
    )["status"] == "ok"
    assert store.open_stage(
        ROUND_ID, 1, participants, list(range(10))
    )["assignments"] == 80

    failure_times = {
        (2, 1): "2026-09-12T00:13:21.997400Z",
        (2, 2): "2026-09-12T00:17:22.014805Z",
        (4, 1): "2026-09-12T00:19:12.426582Z",
        (4, 2): "2026-09-12T00:22:37.787573Z",
        (5, 1): "2026-09-12T00:21:26.477274Z",
        (5, 2): "2026-09-12T00:22:37.617830Z",
        (6, 1): "2026-09-12T00:22:37.656981Z",
        (6, 2): "2026-09-12T00:22:49.858320Z",
        (7, 1): "2026-09-12T00:22:58.011208Z",
        (7, 2): "2026-09-12T00:24:23.849107Z",
        (8, 1): "2026-09-12T00:24:58.349118Z",
        (8, 2): "2026-09-12T00:25:56.494316Z",
        (9, 1): "2026-09-12T00:27:49.127632Z",
        (9, 2): "2026-09-12T00:29:15.689454Z",
    }
    failed_positions = {2, 4, 5, 6, 7, 8, 9}
    with connection.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds "
            "DISABLE TRIGGER lab_arena_rounds_write_once"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc = "
            "pg_catalog.jsonb_set(configuration_doc, '{schedule}', %s::jsonb) "
            "WHERE round_id = %s",
            (json.dumps(stored_schedule), ROUND_ID),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds "
            "ENABLE TRIGGER lab_arena_rounds_write_once"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs "
            "DISABLE TRIGGER lab_arena_runs_terminal"
        )
        for submission_id in PARTICIPANT_IDS:
            for position in range(10):
                run_id = f"{ROUND_ID}:{submission_id}:1:{position}:1"
                cause = (
                    "model_error" if submission_id == FAILED_SUBMISSION
                    and position in {2, 4, 5}
                    else "provider_error" if submission_id == FAILED_SUBMISSION
                    and position in {6, 7, 8, 9}
                    else "accepted"
                )
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET status = %s, "
                    "terminal_cause = %s, result_doc = %s::jsonb, output_ref = %s, "
                    "runner_hotkey = %s, updated_at = %s WHERE run_id = %s",
                    (
                        "accepted" if cause == "accepted" else "failed",
                        cause,
                        json.dumps({"terminal_status": cause, "fixture": run_id}),
                        f"arena/{ROUND_ID}/outputs/{run_id}.json"
                        if cause == "accepted" else None,
                        RUNNER,
                        failure_times.get((position, 1),
                                          "2026-09-12T00:20:00Z"),
                        run_id,
                    ),
                )
        for position in sorted(failed_positions):
            assignment_id = f"{ROUND_ID}:{FAILED_SUBMISSION}:1:{position}"
            run_id = f"{assignment_id}:2"
            cursor.execute(
                "INSERT INTO public.lab_arena_runs "
                "(run_id, assignment_id, round_id, submission_id, miner_hotkey, "
                "stage, icp_position, attempt, kind, status, runner_hotkey, "
                "stage_generation, result_doc, terminal_cause, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, 1, %s, 2, 'execute', 'failed', "
                "%s, 1, %s::jsonb, 'provider_error', %s, %s)",
                (
                    run_id, assignment_id, ROUND_ID, FAILED_SUBMISSION,
                    miner_hotkeys[FAILED_SUBMISSION], position, RUNNER,
                    json.dumps({"terminal_status": "provider_error",
                                "fixture": run_id}),
                    "2026-09-12T00:13:21.998078Z",
                    failure_times[(position, 2)],
                ),
            )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs "
            "ENABLE TRIGGER lab_arena_runs_terminal"
        )
        proof_run = f"{ROUND_ID}:{FAILED_SUBMISSION}:1:6:1"
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, "
            "call_identity, provider, operation_id, funding_source, "
            "amount_microusd, entry_doc, created_at) VALUES "
            "('uncertain', %s, %s, %s, %s, 1, %s, 'deepline', "
            "'deepline.execute', 'miner_key', 46218434, %s::jsonb, %s)",
            (
                miner_hotkeys[FAILED_SUBMISSION], ROUND_ID, FAILED_SUBMISSION,
                proof_run, sha("219-402"),
                json.dumps({"call": {"reason": "missing_provider_cost",
                                      "provider_status": 402}}),
                CREDENTIAL_FAILURE_AT,
            ),
        )
        for index, run_id in enumerate(REPAIR_RUN_IDS):
            if run_id == proof_run:
                continue
            reason = "per_icp_quota" if wrong_refusal and index == 0 \
                else "provider_cost_uncertain"
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_kind, miner_hotkey, round_id, submission_id, run_id, "
                "stage, call_identity, provider, operation_id, funding_source, "
                "amount_microusd, entry_doc, created_at) VALUES "
                "('refusal', %s, %s, %s, %s, 1, %s, 'openrouter', "
                "'openrouter.chat', 'miner_key', 0, %s::jsonb, "
                "%s::timestamptz + (%s || ' seconds')::interval)",
                (
                    miner_hotkeys[FAILED_SUBMISSION], ROUND_ID,
                    FAILED_SUBMISSION, run_id, sha(f"219-refusal-{index}"),
                    json.dumps({"reason": reason}), CREDENTIAL_FAILURE_AT,
                    index + 1,
                ),
            )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, "
            "call_identity, provider, operation_id, funding_source, "
            "amount_microusd, entry_doc, terminal_response, created_at) VALUES "
            "('settlement', %s, %s, %s, %s, 1, %s, 'openrouter', "
            "'openrouter.chat', 'miner_key', 3761517, '{}'::jsonb, "
            "'{\"status\":200}'::jsonb, '2026-09-12T00:21:00Z')",
            (
                miner_hotkeys[PRESERVED_SUBMISSION], ROUND_ID,
                PRESERVED_SUBMISSION,
                f"{ROUND_ID}:{PRESERVED_SUBMISSION}:1:0:1",
                sha("219-preserved-settlement"),
            ),
        )
        if live_call:
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_kind, miner_hotkey, round_id, submission_id, run_id, "
                "stage, call_identity, provider, operation_id, funding_source, "
                "amount_microusd, entry_doc, created_at) VALUES "
                "('reservation', %s, %s, %s, %s, 1, %s, 'openrouter', "
                "'openrouter.chat', 'miner_key', 1, '{}'::jsonb, "
                "'2026-09-12T00:30:00Z')",
                (
                    miner_hotkeys[PRESERVED_SUBMISSION], ROUND_ID,
                    PRESERVED_SUBMISSION,
                    f"{ROUND_ID}:{PRESERVED_SUBMISSION}:1:0:1",
                    sha("219-live-call"),
                ),
            )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status = 'cancelled', "
            "cancel_reason = 'execution_incomplete:stage1:4', "
            "published_at = %s WHERE round_id = %s",
            ("2026-09-12T00:32:10Z" if publication_started else None,
             ROUND_ID),
        )
    return connection, store, transport


def test_recovery_preserves_state_and_resumes_canonical_scoring(database):
    connection, store, transport = _seed(database)
    try:
        with connection.cursor() as cursor:
            before = {
                "submissions": _row_hash(cursor, "lab_arena_submissions",
                                         "round_id = %s", (ROUND_ID,)),
                "credentials": _row_hash(
                    cursor, "lab_arena_submission_credentials",
                    "submission_id IN (SELECT submission_id FROM "
                    "public.lab_arena_submissions WHERE round_id = %s)",
                    (ROUND_ID,),
                ),
                "accepted": _row_hash(
                    cursor, "lab_arena_runs",
                    "round_id = %s AND status = 'accepted'", (ROUND_ID,),
                ),
                "ledger": _row_hash(cursor, "lab_arena_ledger",
                                    "round_id = %s", (ROUND_ID,)),
            }
        _apply(connection)
        row = store.get_round(ROUND_ID)
        assert row["status"] == "stage1_closed"
        assert row["cancel_reason"] is None
        execute_runs = store.list_runs(ROUND_ID, stage=1, kind="execute")
        assert len(execute_runs) == 87
        assert sum(run["status"] == "accepted" for run in execute_runs) == 73
        repaired = {run["run_id"]: run for run in execute_runs
                    if run["run_id"] in REPAIR_RUN_IDS}
        assert set(repaired) == set(REPAIR_RUN_IDS)
        assert all(run["terminal_cause"] == "credential_error"
                   and run["result_doc"]["terminal_status"] == "credential_error"
                   for run in repaired.values())
        pre_402 = next(
            run for run in execute_runs
            if run["run_id"] == f"{ROUND_ID}:{FAILED_SUBMISSION}:1:2:2"
        )
        assert pre_402["terminal_cause"] == "provider_error"
        assert len([run for run in execute_runs if run["attempt"] == 2]) == 7

        with connection.cursor() as cursor:
            assert _row_hash(cursor, "lab_arena_submissions",
                             "round_id = %s", (ROUND_ID,)) == before["submissions"]
            assert _row_hash(
                cursor, "lab_arena_submission_credentials",
                "submission_id IN (SELECT submission_id FROM "
                "public.lab_arena_submissions WHERE round_id = %s)",
                (ROUND_ID,),
            ) == before["credentials"]
            assert _row_hash(cursor, "lab_arena_runs",
                             "round_id = %s AND status = 'accepted'",
                             (ROUND_ID,)) == before["accepted"]
            assert _row_hash(cursor, "lab_arena_ledger",
                             "round_id = %s", (ROUND_ID,)) == before["ledger"]
            cursor.execute(
                "SELECT configuration_doc -> 'schedule', "
                "configuration_doc -> 'runner_hotkeys' "
                "FROM public.lab_arena_rounds WHERE round_id = %s",
                (ROUND_ID,),
            )
            schedule, runners = cursor.fetchone()
            assert schedule["stage_1_close"] == "2026-09-12T04:30:01Z"
            assert runners == [RUNNER]
            cursor.execute(
                "SELECT tgname, tgenabled FROM pg_catalog.pg_trigger "
                "WHERE (tgrelid = 'public.lab_arena_runs'::regclass "
                "AND tgname = 'lab_arena_runs_terminal') OR "
                "(tgrelid = 'public.lab_arena_rounds'::regclass "
                "AND tgname = 'lab_arena_rounds_write_once') "
                "ORDER BY tgname"
            )
            assert cursor.fetchall() == [
                ("lab_arena_rounds_write_once", "O"),
                ("lab_arena_runs_terminal", "O"),
            ]

        plan = scoring.build_scoring_plan(
            round_id=ROUND_ID,
            stage=1,
            runs=execute_runs,
        )
        assert len(plan["work_items"]) == 73
        assert len(plan["zero_rows"]) == 7
        zero_causes = {
            row["icp_position"]: row["cause"] for row in plan["zero_rows"]
        }
        assert zero_causes == {
            2: "model_error",
            4: "credential_error",
            5: "credential_error",
            6: "credential_error",
            7: "credential_error",
            8: "credential_error",
            9: "credential_error",
        }
        assert store.transition_round(
            ROUND_ID, "stage1_closed", "stage1_closed",
            {"stage1_scoring_plan_doc": plan},
        )["status"] == "ok"
        opened = store.open_scoring(ROUND_ID, 1, plan["work_items"])
        assert opened["status"] == "ok" and opened["assignments"] == 73
        completed = 0
        while True:
            response, token, _, _ = claim(
                store, ROUND_ID, RUNNER, parallelism=100, ceiling=100
            )
            if response["status"] != "leased":
                break
            assert response["kind"] == "score"
            result = complete(
                store,
                response["run_id"],
                hash_lease_token(token),
                "accepted",
                output_ref=f"arena/{ROUND_ID}/scores/{response['run_id']}.json",
            )
            assert result["status"] == "accepted"
            completed += 1
        assert completed == 73
        assert store.close_scoring(ROUND_ID, 1)["round_status"] == "stage1_judged"
        scores = [{
            "run_id": run["run_id"],
            "per_icp_score": 50 if run["status"] == "accepted" else 0,
        } for run in execute_runs]
        assert store.record_run_scores(ROUND_ID, 1, scores)["recorded"] == 87

        before_replay = store.get_round(ROUND_ID)
        _apply(connection)
        after_replay = store.get_round(ROUND_ID)
        assert after_replay == before_replay
        assert len(store.list_runs(ROUND_ID, stage=1, kind="execute")) == 87
        assert len(store.list_runs(ROUND_ID, stage=1, kind="score")) == 73
    finally:
        connection.close()
        transport.close()


def test_recovery_is_noop_when_round_is_absent(database):
    _reset(database)
    connection = _connect(database)
    connection.autocommit = True
    try:
        _apply(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_rounds "
                "WHERE round_id = %s", (ROUND_ID,),
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("variant", "expected_error"),
    [
        ({"wrong_refusal": True}, "run lacks credential failure proof"),
        ({"live_call": True}, "recovery preflight differs"),
        ({"publication_started": True}, "recovery preflight differs"),
    ],
)
def test_recovery_rejects_drift(database, variant, expected_error):
    connection, _store, transport = _seed(database, **variant)
    try:
        _apply(connection, error=expected_error)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status, cancel_reason FROM public.lab_arena_rounds "
                "WHERE round_id = %s", (ROUND_ID,),
            )
            assert cursor.fetchone() == (
                "cancelled", "execution_incomplete:stage1:4"
            )
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE run_id = ANY(%s) AND terminal_cause = 'provider_error'",
                (list(REPAIR_RUN_IDS),),
            )
            assert cursor.fetchone()[0] == 10
    finally:
        connection.close()
        transport.close()
