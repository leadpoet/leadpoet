"""PostgreSQL proof for the September 14 fixed-scorer recovery."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import contracts
from tests.lab_arena.arena_20260912_recovery_postgres_test import _row_hash
from tests.lab_arena.arena_20260914_remaining_scoring_recovery_postgres_test import (
    MISSING_POSITIONS,
    ORIGINAL_SCHEDULE,
    RECOVERY_SUFFIX as PRIOR_SUFFIX,
    REMAINING_SUBMISSION,
    _bindings,
    _prepare_current,
)
from tests.lab_arena.arena_20260914_scoring_recovery_postgres_test import (
    JUDGE_ONE,
    JUDGE_TWO,
    OLD_DIGEST,
    PARTICIPANT_IDS,
    ROUND_ID,
    UNRELATED_ROUND_ID,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim


ROOT = Path(__file__).resolve().parents[2]
PRIOR_MIGRATION = ROOT / "scripts" / "249-recover-arena-2026-09-14-remaining-scoring.sql"
PROMOTION_MIGRATION = ROOT / "scripts" / "251-lab-arena-twenty-icp-promotion.sql"
MIGRATION = ROOT / "scripts" / "252-recover-arena-2026-09-14-current-scorer.sql"
RECOVERY_SUFFIX = ":recovery252"
FROZEN_DIGEST = "sha256:748e08acbbdf55bbc46f8d510fe9404d4ed31e575bb78f6cd56f32d4caf849bf"
NEW_DIGEST = "sha256:" + "2" * 64
NEW_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
    f"sourcing-model@{NEW_DIGEST}"
)
PRODUCTION_SCHEDULE = {
    **ORIGINAL_SCHEDULE,
    "stage_1_scoring_close": "2026-09-14T22:20:07.492848Z",
    "stage_2_start": "2026-09-14T22:20:08.492848Z",
    "stage_2_close": "2026-09-15T01:20:08.492848Z",
    "final_scoring_close": "2026-09-15T07:50:08.492848Z",
    "stage_3_start": "2026-09-15T07:50:09.492848Z",
    "stage_3_close": "2026-09-15T08:50:09.492848Z",
    "stage_3_scoring_close": "2026-09-15T10:40:09.492848Z",
    "publication_deadline": "2026-09-15T10:40:10.492848Z",
}


@pytest.fixture(scope="module")
def database():
    promotion = CURRENT_SERVICE_MIGRATIONS.index(
        "251-lab-arena-twenty-icp-promotion.sql"
    )
    pre_promotion = CURRENT_SERVICE_MIGRATIONS[:promotion]
    yield from database_with_lab_arena_migration(pre_promotion)


def _sql_hash(cursor, table: str, where: str, params=()) -> str:
    cursor.execute(
        f"SELECT md5(COALESCE(string_agg(md5(to_jsonb(row_value)::text),'|' "  # noqa: S608
        f"ORDER BY {('entry_id' if table == 'lab_arena_ledger' else 'run_id')}),'')) "
        f"FROM (SELECT * FROM public.{table} WHERE {where}) row_value",
        params,
    )
    return cursor.fetchone()[0]


def _migration_sql(cursor) -> str:
    """Substitute only frozen production constants for the disposable fixture."""

    sql = MIGRATION.read_text(encoding="utf-8").replace(FROZEN_DIGEST, NEW_DIGEST)
    assert (
        "v_expected_credential_hash CONSTANT TEXT :=\n"
        "    'e1834999a30b26c36f7759e9d8d8a628';"
    ) in sql
    cursor.execute(
        "SELECT md5(configuration_doc::text) FROM public.lab_arena_rounds "
        "WHERE round_id=%s",
        (ROUND_ID,),
    )
    values = {
        "6c210f6fdf20d410eaec559955fe0507": cursor.fetchone()[0],
        "ae0ead0f07863cf4caaa39d69f5dd612": _sql_hash(
            cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)
        ),
        "8b7d9b0c7da399fbb4bc7b5893391a1a": _sql_hash(
            cursor, "lab_arena_runs", "round_id=%s AND status='accepted'", (ROUND_ID,)
        ),
    }
    cursor.execute(
        "SELECT md5(COALESCE(string_agg(md5(to_jsonb(row_value)::text),'|' "
        "ORDER BY entry_id),'')),count(*),COALESCE(max(entry_id),0) "
        "FROM (SELECT * FROM public.lab_arena_ledger WHERE round_id=%s) row_value",
        (ROUND_ID,),
    )
    ledger_hash, ledger_count, ledger_max = cursor.fetchone()
    values["c81e427f8afdb076aeff6b3bdf7b82a4"] = ledger_hash
    cursor.execute(
        "SELECT md5(string_agg(md5(to_jsonb(row_value)::text),'|' "
        "ORDER BY submission_id)) FROM (SELECT * FROM public.lab_arena_submissions "
        "WHERE round_id=%s) row_value",
        (ROUND_ID,),
    )
    values["939df0f1d86a931feb384115b81e3bed"] = cursor.fetchone()[0]
    cursor.execute(
        "SELECT md5(string_agg(md5(to_jsonb(row_value)::text),'|' "
        "ORDER BY submission_id,provider)) FROM (SELECT * FROM "
        "public.lab_arena_submission_credentials WHERE submission_id=ANY(%s)) row_value",
        (list(PARTICIPANT_IDS),),
    )
    values["e1834999a30b26c36f7759e9d8d8a628"] = cursor.fetchone()[0]
    cursor.execute(
        "SELECT md5(COALESCE(string_agg(md5(to_jsonb(row_value)::text),'|' "
        "ORDER BY cache_key),'')),count(*) FROM (SELECT * FROM "
        "public.lab_arena_judgment_cache WHERE scope_doc->>'round_id'=%s) row_value",
        (ROUND_ID,),
    )
    cache_hash, cache_count = cursor.fetchone()
    values["9488bb4df1ac2e0e4b2e66848dfeca16"] = cache_hash
    for old, new in values.items():
        sql = sql.replace(old, new)
    sql = sql.replace("<> 12315", f"<> {ledger_count}")
    sql = sql.replace("<> 350351", f"<> {ledger_max}")
    sql = sql.replace("<> 56", f"<> {cache_count}")
    prior = _bindings(cursor, PRIOR_SUFFIX)
    for position, _scored, cache_key, input_hash, _scope, _refs in prior:
        production = {
            5: (
                "sha256:ba498c4b7b9ee0a5d88d6a1db36b391ae157407dd91f5b0ed1f694ac4c291f77",
                "sha256:1cb539d3dc9a882b80d37b12f3668bf7832cb6ac527b2e3ef2a8162a4422b0e0",
            ),
            6: (
                "sha256:1b62c23c84742552d518183fcb54a71b1cd475ed78b056189926d76d710b5498",
                "sha256:a0c4667c4694b8e616ef90abed01c35e970a4a76918003a8bf473bf6ebcdf66e",
            ),
            8: (
                "sha256:1e85fe814989df84bf1ec29ef820288d093187d926ac35c01e3782f3db83cdc6",
                "sha256:d15d000e5f5c738c520789573718742a1fd2c36aa613251789f153f4367f2f51",
            ),
        }[position]
        sql = sql.replace(production[0], input_hash).replace(production[1], cache_key)
    return sql


def _prepare_terminal(database):
    # The historical fixture builds a pre-251 round. Current Python no longer
    # prepares the retired confirmation bank, so suspend that old guard only
    # while the historical rows are assembled.
    psycopg2, dsn = database
    bootstrap = psycopg2.connect(**dsn)
    bootstrap.autocommit = True
    with bootstrap.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
            "lab_arena_integrity_round_guard"
        )
    bootstrap.close()
    connection, store, transport = _prepare_current(
        database, prepare_confirmation=False
    )
    with connection.cursor() as cursor:
        cursor.execute(PRIOR_MIGRATION.read_text(encoding="utf-8"))
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal"
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',runner_hotkey=%s,"
            "lease_generation=1,result_doc=jsonb_build_object('terminal_status',"
            "CASE WHEN icp_position=5 THEN 'judge_error' ELSE 'stage_closed' END),"
            "terminal_cause=CASE WHEN icp_position=5 THEN 'judge_error' "
            "ELSE 'stage_closed' END,terminal_doc=jsonb_build_object('fixture',"
            "'recovery249') WHERE round_id=%s AND assignment_id LIKE "
            "'%%:recovery249' AND attempt=1",
            (JUDGE_ONE, ROUND_ID),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
            "submission_id,miner_hotkey,stage,icp_position,attempt,status,"
            "runner_hotkey,previous_runner_hotkey,lease_generation,"
            "stage_generation,kind,scored_run_id,result_doc,terminal_cause,"
            "terminal_doc,judgment_cache_key,judgment_input_hash,"
            "judgment_scope_doc,judgment_group_leader,"
            "judgment_group_miner_hotkeys,company_judgment_refs) "
            "SELECT assignment_id||':2',assignment_id,round_id,submission_id,"
            "miner_hotkey,stage,icp_position,2,'failed',%s,%s,1,stage_generation,"
            "kind,scored_run_id,jsonb_build_object('terminal_status','judge_error'),"
            "'judge_error',jsonb_build_object('fixture','recovery249_retry'),"
            "judgment_cache_key,judgment_input_hash,judgment_scope_doc,"
            "judgment_group_leader,judgment_group_miner_hotkeys,company_judgment_refs "
            "FROM public.lab_arena_runs WHERE round_id=%s AND assignment_id LIKE "
            "'%%:recovery249' AND icp_position=5 AND attempt=1",
            (JUDGE_TWO, JUDGE_ONE, ROUND_ID),
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
            "status_generation=10,stage_generation=9,"
            "cancel_reason='scoring_incomplete',configuration_doc=jsonb_set("
            "configuration_doc,'{schedule}',%s::jsonb,false) WHERE round_id=%s",
            (json.dumps(PRODUCTION_SCHEDULE), ROUND_ID),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
            "lab_arena_rounds_write_once"
        )
        cursor.execute(PROMOTION_MIGRATION.read_text(encoding="utf-8"))
    return connection, store, transport


def test_recovery_rebinds_only_three_and_preserves_history(database):
    connection, store, transport = _prepare_terminal(database)
    try:
        with connection.cursor() as cursor:
            old_runs = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
            old_accepted = _row_hash(
                cursor, "lab_arena_runs", "round_id=%s AND status='accepted'", (ROUND_ID,)
            )
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
            old_bindings = _bindings(cursor, PRIOR_SUFFIX)
            cursor.execute(
                "SELECT configuration_doc-'schedule'-'scorer_image_digest'"
                "-'scorer_image_reference' FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ROUND_ID,),
            )
            old_policy = cursor.fetchone()[0]
            sql = _migration_sql(cursor)
            before = datetime.now(timezone.utc)
            cursor.execute(sql)
            after = datetime.now(timezone.utc)

            assert _row_hash(
                cursor,
                "lab_arena_runs",
                "round_id=%s AND assignment_id NOT LIKE '%%:recovery252'",
                (ROUND_ID,),
            ) == old_runs
            assert _row_hash(
                cursor, "lab_arena_runs", "round_id=%s AND status='accepted'", (ROUND_ID,)
            ) == old_accepted
            assert _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,)) == old_ledger
            assert _row_hash(cursor, "lab_arena_submissions", "round_id=%s", (ROUND_ID,)) == old_submissions
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
            assert _row_hash(cursor, "lab_arena_rounds", "round_id=%s", (UNRELATED_ROUND_ID,)) == old_unrelated

            new_bindings = _bindings(cursor, RECOVERY_SUFFIX)
            assert [row[0] for row in new_bindings] == list(MISSING_POSITIONS)
            for old, new in zip(old_bindings, new_bindings, strict=True):
                assert new[1] == old[1]
                assert new[3] == old[3]
                assert new[5] == old[5]
                assert new[2] != old[2]
                assert new[4]["scorer_image_digest"] == NEW_DIGEST
                assert new[4]["scorer_image_reference"] == NEW_REFERENCE
                assert new[4]["cache_key"] == new[2]
                assert contracts.document_hash(
                    {key: value for key, value in new[4].items() if key != "cache_key"}
                ) == new[2]

            cursor.execute(
                "SELECT status,status_generation,stage_generation,cancel_reason,"
                "configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND_ID,),
            )
            status, status_generation, stage_generation, reason, config = cursor.fetchone()
            assert (status, status_generation, stage_generation, reason) == (
                "stage1_scoring",
                11,
                10,
                None,
            )
            assert config["scorer_image_digest"] == NEW_DIGEST
            assert config["scorer_image_reference"] == NEW_REFERENCE
            assert {
                key: value
                for key, value in config.items()
                if key not in {"schedule", "scorer_image_digest", "scorer_image_reference"}
            } == old_policy
            schedule = config["schedule"]
            assert contracts.validate_document(
                schedule, contracts.STAGE_SCHEDULE_FIELDS
            ) == schedule
            assert not {
                "stage_3_start",
                "stage_3_close",
                "stage_3_scoring_close",
            }.intersection(schedule)
            anchor = datetime.fromisoformat(schedule["stage_1_scoring_close"])
            assert before + timedelta(hours=6, minutes=30) <= anchor
            assert anchor <= after + timedelta(hours=6, minutes=30)
            assert datetime.fromisoformat(schedule["stage_2_start"]) - anchor == timedelta(seconds=1)
            assert datetime.fromisoformat(schedule["stage_2_close"]) - anchor == timedelta(hours=3, seconds=1)
            assert datetime.fromisoformat(schedule["final_scoring_close"]) - anchor == timedelta(hours=9, minutes=30, seconds=1)
            assert datetime.fromisoformat(schedule["publication_deadline"]) - anchor == timedelta(hours=9, minutes=30, seconds=2)

        first, _token, _request_id, _request_hash = claim(store, ROUND_ID, JUDGE_ONE)
        assert first["status"] == "leased"
        assert first["assignment_id"].endswith(RECOVERY_SUFFIX)
        with connection.cursor() as cursor:
            before_replay = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
            before_round = _row_hash(cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,))
            cursor.execute(sql)
            assert _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)) == before_replay
            assert _row_hash(cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,)) == before_round
    finally:
        connection.close()
        transport.close()


@pytest.mark.parametrize("mismatch", ("lineage", "partial_marker", "ledger"))
def test_recovery_rolls_back_on_mismatch(database, mismatch):
    connection, _store, transport = _prepare_terminal(database)
    try:
        with connection.cursor() as cursor:
            sql = _migration_sql(cursor)
            if mismatch == "lineage":
                cursor.execute(
                    "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal"
                )
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET terminal_cause='stage_closed' "
                    "WHERE run_id=%s",
                    (f"{ROUND_ID}:{REMAINING_SUBMISSION}:1:5:score:recovery249:2",),
                )
                cursor.execute(
                    "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal"
                )
            elif mismatch == "partial_marker":
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                    "submission_id,miner_hotkey,stage,icp_position,attempt,status,"
                    "lease_generation,stage_generation,kind,scored_run_id,"
                    "judgment_cache_key,judgment_input_hash,judgment_scope_doc,"
                    "judgment_group_leader,judgment_group_miner_hotkeys) SELECT "
                    "%s,%s,round_id,submission_id,miner_hotkey,stage,icp_position,1,"
                    "'pending',0,10,kind,scored_run_id,judgment_cache_key,"
                    "judgment_input_hash,judgment_scope_doc,true,"
                    "judgment_group_miner_hotkeys FROM public.lab_arena_runs "
                    "WHERE run_id=%s",
                    (
                        f"{ROUND_ID}:{REMAINING_SUBMISSION}:1:5:score:recovery252:1",
                        f"{ROUND_ID}:{REMAINING_SUBMISSION}:1:5:score:recovery252",
                        f"{ROUND_ID}:{REMAINING_SUBMISSION}:1:5:score:recovery249:2",
                    ),
                )
            else:
                cursor.execute(
                    "ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER "
                    "lab_arena_ledger_append_only"
                )
                cursor.execute(
                    "UPDATE public.lab_arena_ledger SET entry_doc=entry_doc||"
                    "'{\"fixture_mismatch\":true}'::jsonb WHERE entry_id=(SELECT "
                    "max(entry_id) FROM public.lab_arena_ledger WHERE round_id=%s)",
                    (ROUND_ID,),
                )
                cursor.execute(
                    "ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER "
                    "lab_arena_ledger_append_only"
                )
            before_runs = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
            before_round = _row_hash(cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,))
        with pytest.raises(connection.Error, match="recovery252"):
            with connection.cursor() as cursor:
                cursor.execute(sql)
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
            assert _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)) == before_runs
            assert _row_hash(cursor, "lab_arena_rounds", "round_id=%s", (ROUND_ID,)) == before_round
    finally:
        connection.close()
        transport.close()
