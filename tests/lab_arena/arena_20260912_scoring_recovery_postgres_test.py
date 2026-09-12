"""Exact recovery of the second arena-2026-09-12 cancellation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import hash_lease_token
from tests.lab_arena.arena_20260912_recovery_postgres_test import (
    PARTICIPANT_IDS,
    PRESERVED_SUBMISSION,
    ROUND_ID,
    RUNNER,
    _connect,
    _row_hash,
    _seed,
)
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, hotkey, sha


TARGET = PRESERVED_SUBMISSION
FINALISTS = [
    "sub-c2ab2d55f187807c2793433701485825",
    TARGET,
    "sub-67fdb5d43bc8f9ea4cb6e74df4797037",
    "sub-1460c4e33f19f923b969ef2fd941072f",
    "sub-eb8a764bc501efb7d47f9c6c74d49756",
    "sub-64750217950f089a3bec21ee6fe72d6e",
]
PRODUCTION_PLAN_MD5 = "b2c9fb27a2045cd02c45faa528304037"
PRODUCTION_SCORER = (
    "sha256:1e50937e35c4fe7f099a241c20f0f12f3cb70453d35669d990a968d0734b5631"
)
MIGRATION_219 = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "219-recover-arena-2026-09-12-credential-failure.sql"
)
MIGRATION_220 = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "220-recover-arena-2026-09-12-scoring.sql"
)
MIGRATION_221 = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "221-lab-arena-participation-original-judgments.sql"
)
OLD_ASSIGNMENT = f"{ROUND_ID}:{TARGET}:2:12:score"
TARGET_RUN = f"{OLD_ASSIGNMENT}:2"
REQUEST_HASH = sha("220-identity")
OLD_CALL_IDENTITY = contracts.provider_call_identity(
    assignment_id=OLD_ASSIGNMENT,
    attempt=2,
    icp_position=12,
    action_sequence=0,
    operation_id="openrouter.chat",
    request_hash=REQUEST_HASH,
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _apply_220(connection, plan_md5: str, scorer_digest: str) -> None:
    sql = MIGRATION_220.read_text(encoding="utf-8")
    assert sql.count(PRODUCTION_PLAN_MD5) == 1
    assert sql.count(PRODUCTION_SCORER) == 1
    sql = sql.replace(PRODUCTION_PLAN_MD5, plan_md5)
    sql = sql.replace(PRODUCTION_SCORER, scorer_digest)
    with connection.cursor() as cursor:
        cursor.execute(sql)


def _insert_run(cursor, *, run_id, assignment_id, submission_id, miner_hotkey,
                stage, position, attempt, kind, status, cause,
                scored_run_id=None, previous_runner=None):
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id, assignment_id, round_id, submission_id, miner_hotkey, "
        "stage, icp_position, attempt, kind, scored_run_id, status, "
        "runner_hotkey, previous_runner_hotkey, stage_generation, "
        "lease_generation, lease_token_hash, result_doc, output_ref, "
        "terminal_cause, terminal_doc) VALUES "
        "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,8,2,%s,%s::jsonb,%s,%s,%s::jsonb)",
        (
            run_id, assignment_id, ROUND_ID, submission_id, miner_hotkey,
            stage, position, attempt, kind, scored_run_id, status,
            RUNNER if status != "failed" else None, previous_runner,
            hash_lease_token("pre-recovery-token") if run_id == TARGET_RUN else None,
            json.dumps({"terminal_status": cause, "fixture": run_id}),
            f"arena/{ROUND_ID}/objects/{run_id}.json" if status == "accepted" else None,
            cause,
            json.dumps({"closed_at": "2026-09-12T02:45:46Z", "previous_status": "pending"}),
        ),
    )


def _seed_scoring_failure(database):
    connection, store, transport = _seed(database)
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION_219.read_text(encoding="utf-8"))
        cursor.execute(
            "SELECT submission_id, miner_hotkey FROM public.lab_arena_submissions "
            "WHERE round_id=%s", (ROUND_ID,),
        )
        miners = dict(cursor.fetchall())

        accepted_stage1 = store.list_runs(ROUND_ID, stage=1, kind="execute")
        accepted_stage1 = [r for r in accepted_stage1 if r["status"] == "accepted"]
        for index, execution in enumerate(accepted_stage1):
            assignment = (
                f"{ROUND_ID}:{execution['submission_id']}:1:"
                f"{execution['icp_position']}:score"
            )
            first_failed = index < 7
            _insert_run(
                cursor, run_id=f"{assignment}:1", assignment_id=assignment,
                submission_id=execution["submission_id"],
                miner_hotkey=execution["miner_hotkey"], stage=1,
                position=execution["icp_position"], attempt=1, kind="score",
                status="failed" if first_failed else "accepted",
                cause="judge_error" if first_failed else "accepted",
                scored_run_id=execution["run_id"],
            )
            if first_failed:
                retry_accepted = index < 4
                _insert_run(
                    cursor, run_id=f"{assignment}:2", assignment_id=assignment,
                    submission_id=execution["submission_id"],
                    miner_hotkey=execution["miner_hotkey"], stage=1,
                    position=execution["icp_position"], attempt=2, kind="score",
                    status="accepted" if retry_accepted else "failed",
                    cause="accepted" if retry_accepted else "credential_error",
                    scored_run_id=execution["run_id"], previous_runner=RUNNER,
                )

        stage2_assignments = [
            (submission_id, position)
            for submission_id in PARTICIPANT_IDS
            for position in range(10, 20)
        ]
        failed = stage2_assignments[:15]
        failure_causes = (["credential_error"] * 10 + ["model_error"] * 3
                          + ["provider_error", "model_timeout"])
        accepted_stage2 = []
        for item_index, (submission_id, position) in enumerate(stage2_assignments):
            assignment = f"{ROUND_ID}:{submission_id}:2:{position}"
            if (submission_id, position) in failed:
                cause = failure_causes[failed.index((submission_id, position))]
                _insert_run(
                    cursor, run_id=f"{assignment}:1", assignment_id=assignment,
                    submission_id=submission_id, miner_hotkey=miners[submission_id],
                    stage=2, position=position, attempt=1, kind="execute",
                    status="failed", cause=cause,
                )
                if item_index < 3:
                    run_id = f"{assignment}:2"
                    _insert_run(
                        cursor, run_id=run_id, assignment_id=assignment,
                        submission_id=submission_id,
                        miner_hotkey=miners[submission_id], stage=2,
                        position=position, attempt=2, kind="execute",
                        status="accepted", cause="accepted", previous_runner=RUNNER,
                    )
                    accepted_stage2.append((submission_id, position, run_id))
            else:
                run_id = f"{assignment}:1"
                _insert_run(
                    cursor, run_id=run_id, assignment_id=assignment,
                    submission_id=submission_id, miner_hotkey=miners[submission_id],
                    stage=2, position=position, attempt=1, kind="execute",
                    status="accepted", cause="accepted",
                )
                accepted_stage2.append((submission_id, position, run_id))
        assert len(accepted_stage2) == 68

        work_items = []
        other_score_keys = sorted(
            (submission_id, position)
            for submission_id, position, _run_id in accepted_stage2
            if (submission_id, position) not in {
                (TARGET, 10), (TARGET, 11), (TARGET, 12), (TARGET, 13)
            }
        )
        accepted_score_keys = set(other_score_keys[:21]) | {
            (TARGET, 10), (TARGET, 11)
        }
        assert len(accepted_score_keys) == 23
        for submission_id, position, scored_run_id in accepted_stage2:
            assignment = f"{ROUND_ID}:{submission_id}:2:{position}:score"
            accepted = (submission_id, position) in accepted_score_keys
            cause = "accepted" if accepted else (
                "judge_error" if submission_id == TARGET and position in (12, 13)
                else "stage_closed"
            )
            _insert_run(
                cursor, run_id=f"{assignment}:1", assignment_id=assignment,
                submission_id=submission_id, miner_hotkey=miners[submission_id],
                stage=2, position=position, attempt=1, kind="score",
                status="accepted" if accepted else "failed", cause=cause,
                scored_run_id=scored_run_id,
            )
            work_items.append({
                "submission_id": submission_id,
                "icp_position": position,
                "scored_run_id": scored_run_id,
                "output_ref": f"arena/{ROUND_ID}/objects/{scored_run_id}.json",
            })
        for position, cause in ((12, "judge_error"), (13, "stage_closed")):
            assignment = f"{ROUND_ID}:{TARGET}:2:{position}:score"
            scored_run_id = next(
                run_id for submission_id, p, run_id in accepted_stage2
                if submission_id == TARGET and p == position
            )
            _insert_run(
                cursor, run_id=f"{assignment}:2", assignment_id=assignment,
                submission_id=TARGET, miner_hotkey=miners[TARGET], stage=2,
                position=position, attempt=2, kind="score", status="failed",
                cause=cause, scored_run_id=scored_run_id, previous_runner=RUNNER,
            )

        zero_rows = [
            {"submission_id": submission_id, "icp_position": position,
             "cause": failure_causes[index]}
            for index, (submission_id, position) in enumerate(failed[3:], start=3)
        ]
        plan = {
            "schema_version": "leadpoet.lab_arena.scoring_plan.v1",
            "round_id": ROUND_ID,
            "stage": 2,
            "work_items": work_items,
            "zero_rows": zero_rows,
        }
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled', "
            "status_generation=11, stage_generation=9, "
            "cancel_reason='scoring_incomplete', finalists=%s::jsonb, "
            "stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s",
            (json.dumps(FINALISTS), json.dumps(plan), ROUND_ID),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )

        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc,terminal_response,created_at) VALUES "
            "(301884,'settlement',%s,%s,%s,%s,2,%s,'deepline',"
            "'scrapingdog.scrape','miner_key',2000,'{}'::jsonb,%s::jsonb,"
            "'2026-09-12T02:44:00.644946Z'),"
            "(301934,'uncertain',%s,%s,%s,%s,2,%s,'deepline',"
            "'scrapingdog.scrape','miner_key',49329183,%s::jsonb,NULL,"
            "'2026-09-12T02:44:33.249367Z'),"
            "(301941,'uncertain',%s,%s,%s,%s,2,%s,'deepline',"
            "'scrapingdog.scrape','miner_key',4217,%s::jsonb,NULL,"
            "'2026-09-12T02:44:39.668624Z')",
            (
                miners[TARGET], ROUND_ID, TARGET,
                f"{ROUND_ID}:{TARGET}:2:11:score:1",
                "sha256:21d55103563b01712ff2033ddcc10c55e23e8712e99c485e94b912760a66d880",
                json.dumps({"provider_cost": {
                    "request_id": "iad1::nrll6-1789181035861-44985ee12c9a"}}),
                miners[TARGET], ROUND_ID, TARGET,
                f"{ROUND_ID}:{TARGET}:2:12:score:1",
                "sha256:6acff63503cff507fd39853485d8bf58a8b8bca6c50ae0a64be1afa774149b74",
                json.dumps({"reason": "worker_reported", "call": {
                    "reason": "missing_provider_cost", "provider_status": 502,
                    "response_provenance": "response_too_large"}}),
                miners[TARGET], ROUND_ID, TARGET,
                f"{ROUND_ID}:{TARGET}:2:12:score:1",
                "sha256:ba4e76b407cf02a5786b35a5ef03c040f3810b60f1c81070c73c6f654a213a92",
                json.dumps({"reason": "worker_reported", "call": {
                    "reason": "missing_provider_cost", "provider_status": 502,
                    "response_provenance": "response_too_large"}}),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger "
            "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc) VALUES ('refusal',%s,%s,%s,%s,2,%s,'openrouter',"
            "'openrouter.chat','miner_key',0,%s::jsonb)",
            (miners[TARGET], ROUND_ID, TARGET, TARGET_RUN, OLD_CALL_IDENTITY,
             json.dumps({"reason": "provider_cost_uncertain"})),
        )
        cursor.execute(
            "SELECT md5(stage2_scoring_plan_doc::text), "
            "configuration_doc->>'scorer_image_digest' "
            "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND_ID,),
        )
        plan_md5, scorer_digest = cursor.fetchone()
    return connection, store, transport, plan_md5, scorer_digest


def test_scoring_recovery_preserves_state_and_uses_fresh_call_identity(database):
    connection, store, transport, plan_md5, scorer_digest = _seed_scoring_failure(database)
    try:
        with connection.cursor() as cursor:
            before = {
                "accepted": _row_hash(cursor, "lab_arena_runs",
                                      "round_id=%s AND status='accepted'", (ROUND_ID,)),
                "submissions": _row_hash(cursor, "lab_arena_submissions",
                                          "round_id=%s", (ROUND_ID,)),
                "credentials": _row_hash(
                    cursor, "lab_arena_submission_credentials",
                    "submission_id=ANY(%s)", (list(PARTICIPANT_IDS),),
                ),
            }
            cursor.execute(
                "SELECT stage,kind,count(*),count(distinct assignment_id),"
                "count(*) filter(where status='accepted' and terminal_cause='accepted'),"
                "count(*) filter(where status='failed' and terminal_cause='judge_error'),"
                "count(*) filter(where status='failed' and terminal_cause='stage_closed') "
                "FROM public.lab_arena_runs WHERE round_id=%s GROUP BY stage,kind ORDER BY stage,kind",
                (ROUND_ID,),
            )
            assert cursor.fetchall() == [
                (1, "execute", 87, 80, 73, 0, 0),
                (1, "score", 80, 73, 70, 7, 0),
                (2, "execute", 83, 80, 68, 0, 0),
                (2, "score", 70, 68, 23, 3, 44),
            ]
        _apply_220(connection, plan_md5, scorer_digest)
        round_row = store.get_round(ROUND_ID)
        assert (round_row["status"], round_row["status_generation"],
                round_row["stage_generation"]) == ("stage2_scoring", 12, 10)

        with connection.cursor() as cursor:
            assert _row_hash(cursor, "lab_arena_runs",
                             "round_id=%s AND status='accepted'", (ROUND_ID,)) == before["accepted"]
            assert _row_hash(cursor, "lab_arena_submissions",
                             "round_id=%s", (ROUND_ID,)) == before["submissions"]
            assert _row_hash(cursor, "lab_arena_submission_credentials",
                             "submission_id=ANY(%s)", (list(PARTICIPANT_IDS),)) == before["credentials"]
            cursor.execute(
                "SELECT count(*),count(distinct assignment_id),"
                "count(*) filter(where status='pending') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND stage=2 AND kind='score'",
                (ROUND_ID,),
            )
            assert cursor.fetchone() == (70, 68, 45)
            cursor.execute(
                "SELECT entry_id,amount_microusd,entry_kind,"
                "entry_doc#>>'{recovery220,original_amount_microusd}' "
                "FROM public.lab_arena_ledger WHERE entry_id IN (301934,301941) "
                "ORDER BY entry_id"
            )
            assert cursor.fetchall() == [
                (301934, 4000, "uncertain", "49329183"),
                (301941, 4000, "uncertain", "4217"),
            ]
            cursor.execute(
                "SELECT tgname,tgenabled FROM pg_catalog.pg_trigger WHERE "
                "(tgrelid='public.lab_arena_runs'::regclass AND tgname='lab_arena_runs_terminal') OR "
                "(tgrelid='public.lab_arena_rounds'::regclass AND tgname='lab_arena_rounds_write_once') OR "
                "(tgrelid='public.lab_arena_ledger'::regclass AND tgname='lab_arena_ledger_append_only') "
                "ORDER BY tgname"
            )
            assert cursor.fetchall() == [
                ("lab_arena_ledger_append_only", "O"),
                ("lab_arena_rounds_write_once", "O"),
                ("lab_arena_runs_terminal", "O"),
            ]

        before_immediate_replay = store.get_round(ROUND_ID)
        _apply_220(connection, plan_md5, scorer_digest)
        assert store.get_round(ROUND_ID) == before_immediate_replay

        stale = store.reserve_call(
            run_id=TARGET_RUN,
            lease_token_hash=hash_lease_token("pre-recovery-token"),
            call_identity=OLD_CALL_IDENTITY, operation_id="openrouter.chat",
            provider="openrouter", funding_source="miner_key",
            amount_microusd=1000, call_doc={},
        )
        assert stale["status"] == "stale"

        completed = 0
        while True:
            response, token, _, _ = claim(
                store, ROUND_ID, RUNNER, parallelism=100, ceiling=100
            )
            if response["run_id"] == TARGET_RUN:
                break
            assert complete(
                store, response["run_id"], hash_lease_token(token), "accepted",
                output_ref=f"arena/{ROUND_ID}/recovered/{response['run_id']}.json",
            )["status"] == "accepted"
            completed += 1
        assert response["run_id"] == TARGET_RUN
        assert response["assignment_id"] == OLD_ASSIGNMENT + ":recovery220"
        token_hash = hash_lease_token(token)
        replay = store.reserve_call(
            run_id=TARGET_RUN, lease_token_hash=token_hash,
            call_identity=OLD_CALL_IDENTITY, operation_id="openrouter.chat",
            provider="openrouter", funding_source="miner_key",
            amount_microusd=1000, call_doc={},
        )
        assert replay["status"] == "refused" and replay["idempotent"] is True
        new_identity = contracts.provider_call_identity(
            assignment_id=response["assignment_id"], attempt=2,
            icp_position=12, action_sequence=0,
            operation_id="openrouter.chat", request_hash=REQUEST_HASH,
        )
        assert new_identity != OLD_CALL_IDENTITY
        assert store.reserve_call(
            run_id=TARGET_RUN, lease_token_hash=token_hash,
            call_identity=new_identity, operation_id="openrouter.chat",
            provider="openrouter", funding_source="miner_key",
            amount_microusd=1000, call_doc={},
        )["status"] == "reserved"
        assert store.mark_dispatched(
            run_id=TARGET_RUN, lease_token_hash=token_hash,
            call_identity=new_identity,
        )["status"] == "dispatched"
        assert store.settle_call(
            run_id=TARGET_RUN, lease_token_hash=token_hash,
            call_identity=new_identity, actual_microusd=500,
            terminal_response={"status": 200},
        )["status"] == "settled"
        assert complete(
            store, TARGET_RUN, token_hash, "accepted",
            output_ref=f"arena/{ROUND_ID}/recovered/{TARGET_RUN}.json",
        )["status"] == "accepted"

        completed += 1
        while True:
            response, token, _, _ = claim(
                store, ROUND_ID, RUNNER, parallelism=100, ceiling=100
            )
            if response["status"] != "leased":
                break
            assert complete(
                store, response["run_id"], hash_lease_token(token), "accepted",
                output_ref=f"arena/{ROUND_ID}/recovered/{response['run_id']}.json",
            )["status"] == "accepted"
            completed += 1
        assert completed == 45
        closed = store.close_scoring(ROUND_ID, 2)
        assert closed["round_status"] == "stage2_judged"
        assert closed["incomplete_assignments"] == 0
        assert len({r["assignment_id"] for r in store.list_runs(
            ROUND_ID, stage=2, kind="score"
        )}) == 68

        before_replay = store.get_round(ROUND_ID)
        _apply_220(connection, plan_md5, scorer_digest)
        assert store.get_round(ROUND_ID) == before_replay
    finally:
        connection.close()
        transport.close()


def test_participation_evidence_survives_recovery_and_replay(database):
    with _connect(database) as setup_connection:
        with setup_connection.cursor() as cursor:
            cursor.execute(MIGRATION_221.read_text(encoding="utf-8"))

    connection, store, transport, plan_md5, scorer_digest = _seed_scoring_failure(
        database
    )
    preserved_run_id = f"{ROUND_ID}:{TARGET}:2:10:score:1"
    preserved_token_hash = hash_lease_token("220-preserved-participation")
    try:
        # The production fixture stores accepted rows directly. Recreate one
        # acceptance through the normal completion RPC so migration 221 records
        # genuine, database-owned participation evidence before recovery 220.
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='stage2_scoring', "
                "stage_generation=8 WHERE round_id=%s",
                (ROUND_ID,),
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
                "lab_arena_runs_terminal"
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='leased', "
                "lease_token_hash=%s, lease_expires_at=clock_timestamp() + "
                "interval '5 minutes', claim_request_id=%s, "
                "claim_request_hash=%s, claim_response='{}'::jsonb, "
                "result_doc=NULL, output_ref=NULL, terminal_cause=NULL, "
                "terminal_doc=NULL WHERE run_id=%s AND status='accepted' "
                "AND participation_accepted_at IS NULL",
                (
                    preserved_token_hash,
                    "1" * 32,
                    sha("220-preserved-participation-request"),
                    preserved_run_id,
                ),
            )
            assert cursor.rowcount == 1
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
                "lab_arena_runs_terminal"
            )
        assert complete(
            store,
            preserved_run_id,
            preserved_token_hash,
            "accepted",
            output_ref=f"arena/{ROUND_ID}/objects/{preserved_run_id}.json",
        )["status"] == "accepted"
        preserved_at = store.get_run(preserved_run_id)[
            "participation_accepted_at"
        ]
        assert preserved_at is not None
        assert store.has_recent_participation("finney", 71, RUNNER)

        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='cancelled', "
                "stage_generation=9 WHERE round_id=%s",
                (ROUND_ID,),
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND participation_accepted_at IS NOT NULL",
                (ROUND_ID,),
            )
            assert cursor.fetchone()[0] == 1

        _apply_220(connection, plan_md5, scorer_digest)
        assert store.get_run(preserved_run_id)[
            "participation_accepted_at"
        ] == preserved_at
        assert store.has_recent_participation("finney", 71, RUNNER)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND participation_accepted_at IS NOT NULL",
                (ROUND_ID,),
            )
            assert cursor.fetchone()[0] == 1
        assert store.get_run(TARGET_RUN)["participation_accepted_at"] is None

        recovered_runner = hotkey("220-fresh-participation-runner")
        assert not store.has_recent_participation("finney", 71, recovered_runner)
        recovered, recovered_token, _, _ = claim(
            store, ROUND_ID, recovered_runner, parallelism=100, ceiling=100
        )
        assert recovered["status"] == "leased"
        assert recovered["assignment_id"].endswith(":recovery220")
        assert store.get_run(recovered["run_id"])[
            "participation_accepted_at"
        ] is None
        recovered_token_hash = hash_lease_token(recovered_token)
        output_ref = f"arena/{ROUND_ID}/recovered/{recovered['run_id']}.json"
        assert complete(
            store,
            recovered["run_id"],
            recovered_token_hash,
            "accepted",
            output_ref=output_ref,
        )["status"] == "accepted"
        recovered_at = store.get_run(recovered["run_id"])[
            "participation_accepted_at"
        ]
        assert recovered_at is not None
        assert store.get_run(recovered["run_id"])["runner_hotkey"] == recovered_runner
        assert store.has_recent_participation("finney", 71, recovered_runner)
        assert store.has_recent_participation("finney", 71, RUNNER)

        replay = complete(
            store,
            recovered["run_id"],
            recovered_token_hash,
            "accepted",
            output_ref=output_ref,
        )
        assert replay["status"] == "accepted" and replay["idempotent"] is True
        assert store.get_run(recovered["run_id"])[
            "participation_accepted_at"
        ] == recovered_at

        _apply_220(connection, plan_md5, scorer_digest)
        assert store.get_run(preserved_run_id)[
            "participation_accepted_at"
        ] == preserved_at
        assert store.get_run(recovered["run_id"])[
            "participation_accepted_at"
        ] == recovered_at
    finally:
        connection.close()
        transport.close()


def test_recovered_attempt_one_keeps_the_frozen_two_attempt_limit(database):
    connection, store, transport, plan_md5, scorer_digest = _seed_scoring_failure(database)
    try:
        _apply_220(connection, plan_md5, scorer_digest)
        response, token, _, _ = claim(
            store, ROUND_ID, RUNNER, parallelism=100, ceiling=100
        )
        assert response["attempt"] == 1
        assert response["assignment_id"].endswith(":recovery220")
        first_id = response["run_id"]
        failed = complete(
            store, first_id, hash_lease_token(token), "judge_error"
        )
        assert failed["confirmation_attempt"] == 2
        retry_id = response["assignment_id"] + ":2"
        retry = store.get_run(retry_id)
        assert retry is not None
        assert retry["assignment_id"] == response["assignment_id"]
        assert retry["attempt"] == 2 and retry["status"] == "pending"

        while True:
            claimed, retry_token, _, _ = claim(
                store, ROUND_ID, RUNNER, parallelism=100, ceiling=100
            )
            if claimed["run_id"] == retry_id:
                break
            assert complete(
                store, claimed["run_id"], hash_lease_token(retry_token),
                "accepted",
                output_ref=f"arena/{ROUND_ID}/recovered/{claimed['run_id']}.json",
            )["status"] == "accepted"
        second = complete(
            store, retry_id, hash_lease_token(retry_token), "judge_error"
        )
        assert second["status"] == "failed"
        assert "confirmation_attempt" not in second
        assert store.get_run(response["assignment_id"] + ":3") is None
        assert store.get_round(ROUND_ID)["configuration_doc"][
            "max_attempts_per_assignment"
        ] == 2
    finally:
        connection.close()
        transport.close()


def test_recovery_exception_rolls_back_mutations_and_guard_state(database):
    connection, store, transport, plan_md5, scorer_digest = _seed_scoring_failure(database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE OR REPLACE FUNCTION public.test_220_forced_failure() "
                "RETURNS trigger LANGUAGE plpgsql AS $$BEGIN "
                "IF NEW.assignment_id LIKE '%:recovery220' THEN "
                "RAISE EXCEPTION 'test_220_forced_failure'; END IF; RETURN NEW; END$$"
            )
            cursor.execute(
                "CREATE TRIGGER test_220_forced_failure BEFORE UPDATE ON "
                "public.lab_arena_runs FOR EACH ROW EXECUTE FUNCTION "
                "public.test_220_forced_failure()"
            )
        with pytest.raises(Exception, match="test_220_forced_failure"):
            _apply_220(connection, plan_md5, scorer_digest)
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
        assert store.get_round(ROUND_ID)["status"] == "cancelled"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT entry_id,amount_microusd FROM public.lab_arena_ledger "
                "WHERE entry_id IN (301934,301941) ORDER BY entry_id"
            )
            assert cursor.fetchall() == [(301934, 49329183), (301941, 4217)]
            cursor.execute(
                "SELECT tgname,tgenabled FROM pg_catalog.pg_trigger WHERE "
                "(tgrelid='public.lab_arena_runs'::regclass AND tgname='lab_arena_runs_terminal') OR "
                "(tgrelid='public.lab_arena_rounds'::regclass AND tgname='lab_arena_rounds_write_once') OR "
                "(tgrelid='public.lab_arena_ledger'::regclass AND tgname='lab_arena_ledger_append_only') "
                "ORDER BY tgname"
            )
            assert cursor.fetchall() == [
                ("lab_arena_ledger_append_only", "O"),
                ("lab_arena_rounds_write_once", "O"),
                ("lab_arena_runs_terminal", "O"),
            ]
            cursor.execute(
                "DROP TRIGGER test_220_forced_failure ON public.lab_arena_runs"
            )
            cursor.execute("DROP FUNCTION public.test_220_forced_failure()")
    finally:
        connection.close()
        transport.close()
