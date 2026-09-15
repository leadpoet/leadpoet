"""PostgreSQL proof for the final three September 14 scoring assignments."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import contracts
from tests.lab_arena.arena_20260912_recovery_postgres_test import _row_hash
from tests.lab_arena.arena_20260914_scoring_recovery_postgres_test import (
    JUDGE_ONE,
    JUDGE_TWO,
    OLD_DIGEST,
    OLD_REFERENCE,
    PARTICIPANT_IDS,
    ROUND_ID,
    UNRELATED_ROUND_ID,
    _prepare,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim


ROOT = Path(__file__).resolve().parents[2]
PRIOR_MIGRATION = ROOT / "scripts" / "244-recover-arena-2026-09-14-scoring.sql"
MIGRATION = ROOT / "scripts" / "249-recover-arena-2026-09-14-remaining-scoring.sql"
MISSING_POSITIONS = (5, 6, 8)
REMAINING_SUBMISSION = "sub-dd76cafd44732cb230a476ab031b24ad"
ACCEPTED_RETRIES = (
    ("sub-09f530ce94b6f63221ce4d69962e3988", 1),
    ("sub-2968d4fba9582f9d8881bfd998290cbe", 9),
)
RECOVERY_SUFFIX = ":recovery249"

ORIGINAL_SCHEDULE = {
    "submission_open": "2026-09-13T00:00:00Z",
    "submission_cutoff": "2026-09-14T00:00:00Z",
    "benchmark_deadline": "2026-09-14T00:30:00Z",
    "stage_1_start": "2026-09-14T00:30:01Z",
    "stage_1_close": "2026-09-14T04:30:01Z",
    "stage_1_scoring_close": "2026-09-14T11:00:01Z",
    "stage_2_start": "2026-09-14T11:00:02Z",
    "stage_2_close": "2026-09-14T14:00:02Z",
    "final_scoring_close": "2026-09-14T20:30:02Z",
    "stage_3_start": "2026-09-14T20:30:03Z",
    "stage_3_close": "2026-09-14T21:30:03Z",
    "stage_3_scoring_close": "2026-09-14T23:20:03Z",
    "publication_deadline": "2026-09-14T23:20:04Z",
}


@pytest.fixture(scope="module")
def database():
    promotion = CURRENT_SERVICE_MIGRATIONS.index(
        "251-lab-arena-twenty-icp-promotion.sql"
    )
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS[:promotion]
    )


def _prepare_current(database, *, prepare_confirmation: bool = True):
    """Build the exact cancelled post-recovery244 run shape."""

    connection, store, transport = _prepare(
        database, prepare_confirmation=prepare_confirmation
    )
    with connection.cursor() as cursor:
        cursor.execute(PRIOR_MIGRATION.read_text(encoding="utf-8"))
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal"
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='accepted',runner_hotkey=%s,"
            "lease_generation=1,result_doc=%s::jsonb,"
            "output_ref='arena/recovered/'||run_id||'.json',"
            "terminal_cause='accepted',terminal_doc=%s::jsonb "
            "WHERE round_id=%s AND assignment_id LIKE '%%:recovery244' "
            "AND NOT (submission_id=%s AND icp_position=ANY(%s)) "
            "AND NOT ((submission_id=%s AND icp_position=%s) OR "
            "(submission_id=%s AND icp_position=%s))",
            (
                JUDGE_ONE,
                json.dumps({"terminal_status": "accepted"}),
                json.dumps({"fixture": "accepted"}),
                ROUND_ID,
                REMAINING_SUBMISSION,
                list(MISSING_POSITIONS),
                ACCEPTED_RETRIES[0][0],
                ACCEPTED_RETRIES[0][1],
                ACCEPTED_RETRIES[1][0],
                ACCEPTED_RETRIES[1][1],
            ),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',runner_hotkey=%s,"
            "lease_generation=1,result_doc=jsonb_build_object('terminal_status',"
            "CASE WHEN icp_position=5 THEN 'judge_error' ELSE 'stage_closed' END),"
            "terminal_cause=CASE WHEN icp_position=5 THEN 'judge_error' "
            "ELSE 'stage_closed' END,terminal_doc=%s::jsonb WHERE round_id=%s "
            "AND assignment_id LIKE '%%:recovery244' AND submission_id=%s "
            "AND icp_position=ANY(%s) AND attempt=1",
            (
                JUDGE_ONE,
                json.dumps({"fixture": "missing"}),
                ROUND_ID,
                REMAINING_SUBMISSION,
                list(MISSING_POSITIONS),
            ),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',runner_hotkey=%s,"
            "lease_generation=1,result_doc=%s::jsonb,"
            "terminal_cause='credential_error',terminal_doc=%s::jsonb "
            "WHERE round_id=%s AND assignment_id LIKE '%%:recovery244' "
            "AND ((submission_id=%s AND icp_position=%s) OR "
            "(submission_id=%s AND icp_position=%s)) AND attempt=1",
            (
                JUDGE_ONE,
                json.dumps({"terminal_status": "credential_error"}),
                json.dumps({"fixture": "credential_error"}),
                ROUND_ID,
                ACCEPTED_RETRIES[0][0],
                ACCEPTED_RETRIES[0][1],
                ACCEPTED_RETRIES[1][0],
                ACCEPTED_RETRIES[1][1],
            ),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET terminal_cause='judge_error',"
            "result_doc=jsonb_build_object('terminal_status','judge_error') "
            "WHERE round_id=%s AND assignment_id NOT LIKE '%%:recovery244' "
            "AND submission_id=%s AND icp_position=ANY(%s) AND kind='score' "
            "AND terminal_cause='stage_closed'",
            (ROUND_ID, REMAINING_SUBMISSION, list(MISSING_POSITIONS)),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
            "submission_id,miner_hotkey,stage,icp_position,attempt,kind,"
            "scored_run_id,status,runner_hotkey,previous_runner_hotkey,"
            "stage_generation,lease_generation,result_doc,terminal_cause,"
            "terminal_doc,judgment_cache_key,judgment_input_hash,"
            "judgment_scope_doc,judgment_group_leader,"
            "judgment_group_miner_hotkeys,company_judgment_refs) "
            "SELECT assignment_id||':2',assignment_id,round_id,submission_id,"
            "miner_hotkey,stage,icp_position,2,kind,scored_run_id,"
            "CASE WHEN submission_id=%s THEN 'failed' ELSE 'accepted' END,%s,%s,"
            "stage_generation,1,jsonb_build_object('terminal_status',"
            "CASE WHEN submission_id=%s THEN 'stage_closed' ELSE 'accepted' END),"
            "CASE WHEN submission_id=%s THEN 'stage_closed' ELSE 'accepted' END,"
            "jsonb_build_object('fixture','attempt_two'),"
            "judgment_cache_key,judgment_input_hash,judgment_scope_doc,"
            "judgment_group_leader,judgment_group_miner_hotkeys,"
            "company_judgment_refs FROM public.lab_arena_runs "
            "WHERE round_id=%s AND assignment_id LIKE '%%:recovery244' "
            "AND ((submission_id=%s AND icp_position=5) OR "
            "(submission_id=%s AND icp_position=%s) OR "
            "(submission_id=%s AND icp_position=%s)) AND attempt=1",
            (
                REMAINING_SUBMISSION,
                JUDGE_TWO,
                JUDGE_ONE,
                REMAINING_SUBMISSION,
                REMAINING_SUBMISSION,
                ROUND_ID,
                REMAINING_SUBMISSION,
                ACCEPTED_RETRIES[0][0],
                ACCEPTED_RETRIES[0][1],
                ACCEPTED_RETRIES[1][0],
                ACCEPTED_RETRIES[1][1],
            ),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET output_ref='arena/recovered/'||run_id||'.json' "
            "WHERE round_id=%s AND assignment_id LIKE '%%:recovery244' "
            "AND attempt=2 AND status='accepted'",
            (ROUND_ID,),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal"
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled',"
            "status_generation=8,stage_generation=7,"
            "cancel_reason='scoring_incomplete',"
            "configuration_doc=jsonb_set(configuration_doc,'{schedule}',"
            "%s::jsonb,false) WHERE round_id=%s",
            (json.dumps(ORIGINAL_SCHEDULE), ROUND_ID),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
    return connection, store, transport


def _bindings(cursor, suffix: str):
    cursor.execute(
        "SELECT DISTINCT ON (icp_position) icp_position,scored_run_id,judgment_cache_key,"
        "judgment_input_hash,judgment_scope_doc,company_judgment_refs "
        "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
        "AND assignment_id LIKE %s AND icp_position=ANY(%s) "
        "ORDER BY icp_position,stage_generation DESC,attempt DESC",
        (ROUND_ID, REMAINING_SUBMISSION, "%" + suffix, list(MISSING_POSITIONS)),
    )
    return cursor.fetchall()


def test_recovery_preserves_history_adds_only_three_and_is_replay_safe(database):
    connection, store, transport = _prepare_current(database)
    try:
        with connection.cursor() as cursor:
            old_runs = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
            old_ledger = _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,))
            old_submissions = _row_hash(
                cursor, "lab_arena_submissions", "round_id=%s", (ROUND_ID,)
            )
            old_credentials = _row_hash(
                cursor,
                "lab_arena_submission_credentials",
                "submission_id=ANY(%s)",
                (list(PARTICIPANT_IDS),),
            )
            old_cache = _row_hash(
                cursor,
                "lab_arena_judgment_cache",
                "scope_doc->>'round_id'=%s",
                (ROUND_ID,),
            )
            old_unrelated = _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (UNRELATED_ROUND_ID,)
            )
            prior_bindings = _bindings(cursor, ":recovery244")
            cursor.execute(
                "SELECT configuration_doc-'schedule' FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ROUND_ID,),
            )
            old_non_schedule = cursor.fetchone()[0]
            cursor.execute(
                "SELECT plan.item->>'submission_id',"
                "(plan.item->>'icp_position')::integer "
                "FROM public.lab_arena_rounds AS round_row "
                "CROSS JOIN LATERAL jsonb_array_elements("
                "round_row.stage1_scoring_plan_doc->'work_items') AS plan(item) "
                "WHERE round_row.round_id=%s AND NOT EXISTS (SELECT 1 "
                "FROM public.lab_arena_runs AS accepted WHERE accepted.round_id=%s "
                "AND accepted.kind='score' AND accepted.status='accepted' "
                "AND accepted.submission_id=plan.item->>'submission_id' "
                "AND accepted.icp_position=(plan.item->>'icp_position')::smallint "
                "AND accepted.scored_run_id=plan.item->>'scored_run_id') "
                "ORDER BY 1,2",
                (ROUND_ID, ROUND_ID),
            )
            assert cursor.fetchall() == [
                (REMAINING_SUBMISSION, position) for position in MISSING_POSITIONS
            ]
            before = datetime.now(timezone.utc)
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            after = datetime.now(timezone.utc)

            assert _row_hash(
                cursor,
                "lab_arena_runs",
                "round_id=%s AND assignment_id NOT LIKE '%%:recovery249'",
                (ROUND_ID,),
            ) == old_runs
            assert _row_hash(
                cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,)
            ) == old_ledger
            assert _row_hash(
                cursor, "lab_arena_submissions", "round_id=%s", (ROUND_ID,)
            ) == old_submissions
            assert _row_hash(
                cursor,
                "lab_arena_submission_credentials",
                "submission_id=ANY(%s)",
                (list(PARTICIPANT_IDS),),
            ) == old_credentials
            assert _row_hash(
                cursor,
                "lab_arena_judgment_cache",
                "scope_doc->>'round_id'=%s",
                (ROUND_ID,),
            ) == old_cache
            assert _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (UNRELATED_ROUND_ID,)
            ) == old_unrelated
            assert _bindings(cursor, RECOVERY_SUFFIX) == prior_bindings

            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),"
                "bool_and(attempt=1 AND status='pending' "
                "AND stage_generation=8 AND run_id=assignment_id||':1') "
                "FROM public.lab_arena_runs WHERE round_id=%s "
                "AND assignment_id LIKE '%%:recovery249'",
                (ROUND_ID,),
            )
            assert cursor.fetchone() == (3, 3, True)
            cursor.execute(
                "SELECT icp_position FROM public.lab_arena_runs "
                "WHERE round_id=%s AND assignment_id LIKE '%%:recovery249' "
                "ORDER BY icp_position",
                (ROUND_ID,),
            )
            assert tuple(row[0] for row in cursor.fetchall()) == MISSING_POSITIONS
            cursor.execute(
                "SELECT status,status_generation,stage_generation,cancel_reason,"
                "configuration_doc-'schedule',configuration_doc->'schedule' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND_ID,),
            )
            (
                status,
                status_generation,
                stage_generation,
                reason,
                non_schedule,
                schedule,
            ) = cursor.fetchone()
            assert (status, status_generation, stage_generation, reason) == (
                "stage1_scoring",
                9,
                8,
                None,
            )
            assert non_schedule == old_non_schedule
            assert contracts.validate_document(
                schedule, contracts.STAGE_SCHEDULE_FIELDS
            ) == schedule
            for field in (
                "submission_open",
                "submission_cutoff",
                "benchmark_deadline",
                "stage_1_start",
                "stage_1_close",
            ):
                assert schedule[field] == ORIGINAL_SCHEDULE[field]

            anchor = datetime.fromisoformat(schedule["stage_1_scoring_close"])
            assert before + timedelta(hours=6, minutes=30) <= anchor
            assert anchor <= after + timedelta(hours=6, minutes=30)
            expected_offsets = {
                "stage_2_start": timedelta(seconds=1),
                "stage_2_close": timedelta(hours=3, seconds=1),
                "final_scoring_close": timedelta(hours=9, minutes=30, seconds=1),
                "stage_3_start": timedelta(hours=9, minutes=30, seconds=2),
                "stage_3_close": timedelta(hours=10, minutes=30, seconds=2),
                "stage_3_scoring_close": timedelta(hours=12, minutes=20, seconds=2),
                "publication_deadline": timedelta(hours=12, minutes=20, seconds=3),
            }
            for field, offset in expected_offsets.items():
                assert datetime.fromisoformat(schedule[field]) - anchor == offset

        first, _token, _request_id, _request_hash = claim(store, ROUND_ID, JUDGE_ONE)
        assert first["status"] == "leased"
        assert first["assignment_id"].endswith(RECOVERY_SUFFIX)
        with connection.cursor() as cursor:
            before_replay_runs = _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)
            )
            before_replay_round = _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,)
            )
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            assert _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)
            ) == before_replay_runs
            assert _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,)
            ) == before_replay_round
    finally:
        connection.close()
        transport.close()


@pytest.mark.parametrize("mismatch", ("run_counts", "partial_marker"))
def test_recovery_rolls_back_on_mismatch(database, mismatch):
    connection, _store, transport = _prepare_current(database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
                "lab_arena_runs_terminal"
            )
            if mismatch == "run_counts":
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET terminal_cause='judge_error' "
                    "WHERE run_id=%s",
                    (
                        f"{ROUND_ID}:{REMAINING_SUBMISSION}:1:5:score:recovery244:2",
                    ),
                )
            else:
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs (run_id,assignment_id,"
                    "round_id,submission_id,miner_hotkey,stage,icp_position,attempt,"
                    "status,lease_generation,stage_generation,kind,scored_run_id,"
                    "judgment_cache_key,judgment_input_hash,judgment_scope_doc,"
                    "judgment_group_leader,judgment_group_miner_hotkeys,"
                    "company_judgment_refs) SELECT %s,%s,round_id,submission_id,"
                    "miner_hotkey,stage,icp_position,1,'pending',0,8,kind,"
                    "scored_run_id,judgment_cache_key,judgment_input_hash,"
                    "judgment_scope_doc,true,judgment_group_miner_hotkeys,"
                    "company_judgment_refs FROM public.lab_arena_runs "
                    "WHERE run_id=%s",
                    (
                        f"{ROUND_ID}:{REMAINING_SUBMISSION}:1:5:score:recovery249:1",
                        f"{ROUND_ID}:{REMAINING_SUBMISSION}:1:5:score:recovery249",
                        f"{ROUND_ID}:{REMAINING_SUBMISSION}:1:5:score:recovery244:2",
                    ),
                )
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER "
                "lab_arena_runs_terminal"
            )
            before_runs = _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)
            )
            before_round = _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,)
            )
        with pytest.raises(connection.Error, match="recovery249"):
            with connection.cursor() as cursor:
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
            assert _row_hash(
                cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)
            ) == before_runs
            assert _row_hash(
                cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,)
            ) == before_round
    finally:
        connection.close()
        transport.close()
