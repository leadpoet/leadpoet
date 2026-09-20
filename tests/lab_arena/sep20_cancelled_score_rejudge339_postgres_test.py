"""Cancelled rerun338 saved-output rejudge template in disposable PostgreSQL."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena import sep20_authority_preserving_rejudge332_postgres_test as rerun332
from tests.lab_arena import sep20_scored_authority_rejudge338_postgres_test as rerun338
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS

ROUND = rerun332.ROUND
ARCHIVE = ROUND + "-r339archive"
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/339-arena-2026-09-20-cancelled-score-rejudge.sql.template"
)
CURRENT_SCORER_DIGEST = rerun338.FINAL_SCORER_DIGEST
CURRENT_SCORER_REFERENCE = rerun338.FINAL_SCORER_REFERENCE
NEW_SCORER_DIGEST = "sha256:" + "7d" * 32
NEW_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + NEW_SCORER_DIGEST
)


@pytest.fixture()
def database():
    yield from rerun332.database.__wrapped__()


def _prepare_cancelled338(connection, harness, monkeypatch):
    rerun338._prepare_scored337(connection, harness, monkeypatch)
    schedule = rerun332._schedule()
    with connection.cursor() as cursor:
        monkeypatch.setattr(rerun338, "NEW_SCORER_DIGEST", CURRENT_SCORER_DIGEST)
        monkeypatch.setattr(rerun338, "NEW_SCORER_REFERENCE", CURRENT_SCORER_REFERENCE)
        rendered, _, _ = rerun338._render338(
            cursor, schedule, harness.objects, monkeypatch
        )
        cursor.execute(rendered)
    connection.commit()
    rerun332._drive_rejudge_cycle(
        harness.service, harness.objects, harness.runner_keys[0]
    )
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT run_id,assignment_id,submission_id,icp_position FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='score' AND stage=1 "
            "AND status='accepted' ORDER BY run_id",
            (ROUND,),
        )
        stage1 = cursor.fetchall()
        assert len(stage1) >= 48
        exhausted = next(row for row in stage1 if row[3] == 8)
        remaining = [row for row in stage1 if row[0] != exhausted[0]]
        accepted = remaining[:37]
        stage_closed = remaining[37:47]
        retained_ids = [row[0] for row in accepted] + [exhausted[0]] + [
            row[0] for row in stage_closed
        ]

        cursor.execute("SET LOCAL session_replication_role=replica")
        # This fixture first drives the full hermetic lifecycle, then recreates
        # the exact terminal classes seen in production. Remove derived cache
        # rows before deleting their source scores; the migration must preserve
        # whatever cache rows exist at its actual sealed preimage.
        cursor.execute("DELETE FROM public.lab_arena_company_judgment_reservations")
        cursor.execute("DELETE FROM public.lab_arena_company_judgments")
        cursor.execute("DELETE FROM public.lab_arena_judgment_cache")
        cursor.execute(
            "DELETE FROM public.lab_arena_ledger l WHERE l.round_id=%s AND EXISTS("
            "SELECT 1 FROM public.lab_arena_runs r WHERE r.run_id=l.run_id "
            "AND r.round_id=%s AND r.kind='score' AND NOT(r.run_id=ANY(%s)))",
            (ROUND, ROUND, retained_ids),
        )
        cursor.execute(
            "DELETE FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' "
            "AND NOT(run_id=ANY(%s))",
            (ROUND, retained_ids),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',"
            "result_doc=jsonb_build_object('terminal_status','judge_error'),output_ref=NULL "
            "WHERE run_id=%s",
            (exhausted[0],),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record("
            "NULL::public.lab_arena_runs,to_jsonb(r)||jsonb_build_object("
            "'run_id',r.assignment_id||':2','attempt',2,'status','failed',"
            "'terminal_cause','judge_error','result_doc',"
            "jsonb_build_object('terminal_status','judge_error'),'output_ref',NULL,"
            "'claim_request_id',NULL,'claim_request_hash',NULL,'claim_response',NULL))).* "
            "FROM public.lab_arena_runs r WHERE r.run_id=%s",
            (exhausted[0],),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',"
            "result_doc=NULL,output_ref=NULL,terminal_doc=jsonb_build_object("
            "'previous_status','pending') WHERE run_id=ANY(%s)",
            ([row[0] for row in stage_closed],),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL "
            "WHERE round_id=%s AND kind='execute'",
            (ROUND,),
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled',"
            "status_generation=status_generation+1,stage_generation=stage_generation+1,"
            "cancel_reason='scoring_incomplete:stage1:1',stage2_scoring_plan_doc=NULL,"
            "finalists=NULL WHERE round_id=%s",
            (ROUND,),
        )
        cursor.execute("SET LOCAL session_replication_role=origin")
        cursor.execute(
            "SELECT count(*) FILTER (WHERE status='accepted'),"
            "count(*) FILTER (WHERE terminal_cause='judge_error'),"
            "count(*) FILTER (WHERE terminal_cause='stage_closed'),count(*) "
            "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
            (ROUND,),
        )
        assert cursor.fetchone() == (37, 2, 10, 49)
    connection.commit()
    return exhausted[2]


def _render339(cursor, schedule, objects, exhausted_submission, tmp_path, monkeypatch):
    values = {
        "__TERMINAL_ACCEPTED_SCORE_COUNT__": "37",
        "__TERMINAL_JUDGE_ERROR_COUNT__": "2",
        "__TERMINAL_STAGE_CLOSED_COUNT__": "10",
        "__TERMINAL_EXHAUSTED_SUBMISSION_SQL__": "'" + exhausted_submission + "'",
    }
    body = TEMPLATE.read_text()
    for marker, value in values.items():
        body = body.replace(marker, value)
    fixture_template = tmp_path / TEMPLATE.name
    fixture_template.write_text(body)
    monkeypatch.setattr(rerun332, "TEMPLATE", fixture_template)
    monkeypatch.setattr(rerun332, "RENDERED", tmp_path / "absent-rendered.sql")
    monkeypatch.setattr(rerun332, "ARCHIVE", ARCHIVE)
    monkeypatch.setattr(
        rerun332,
        "PRIOR_ARCHIVES",
        rerun338.PRIOR_ARCHIVES + (rerun338.ARCHIVE,),
    )
    monkeypatch.setattr(rerun332, "NEW_SCORER_DIGEST", NEW_SCORER_DIGEST)
    monkeypatch.setattr(rerun332, "NEW_SCORER_REFERENCE", NEW_SCORER_REFERENCE)
    return rerun332._render(cursor, schedule, objects)


def test_cancelled338_rejudge_archives_partial_scores_and_preserves_saved_outputs(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        exhausted_submission = _prepare_cancelled338(
            connection, harness, monkeypatch
        )
        with connection.cursor() as cursor:
            authority_before = rerun332._authority_snapshot(cursor)
            execution_before = rerun332._execution_snapshot(cursor)
            score_before = rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(r)-'round_id'-'submission_id' ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            ledger_before = rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s AND EXISTS(SELECT 1 "
                "FROM public.lab_arena_runs r WHERE r.run_id=l.run_id AND r.round_id=%s "
                "AND r.kind='score')",
                (ROUND, ROUND),
            )
            rendered, scoring_before, scoring_after = _render339(
                cursor,
                rerun332._schedule(),
                harness.objects,
                exhausted_submission,
                tmp_path,
                monkeypatch,
            )
            cursor.execute("SAVEPOINT terminal_drift")
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET terminal_cause='judge_timeout' "
                "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='score' AND terminal_cause='stage_closed')",
                (ROUND,),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            cursor.execute("SAVEPOINT refused")
            with pytest.raises(psycopg2.Error, match="terminal preimage differs"):
                cursor.execute(rerun332._migration_body(rendered))
            cursor.execute("ROLLBACK TO SAVEPOINT refused")
            cursor.execute("ROLLBACK TO SAVEPOINT terminal_drift")
            cursor.execute(rendered)
            cursor.execute(rendered)
            assert rerun332._authority_snapshot(cursor) == authority_before
            assert rerun332._execution_snapshot(cursor) == execution_before
            assert rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(r)-'round_id'-'submission_id' ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='score'",
                (ARCHIVE,),
            ) == score_before
            assert rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s AND EXISTS(SELECT 1 "
                "FROM public.lab_arena_runs r WHERE r.run_id=l.run_id AND r.round_id=%s "
                "AND r.kind='score')",
                (ARCHIVE, ARCHIVE),
            ) == ledger_before
            cursor.execute(
                "SELECT status,cancel_reason,configuration_doc->>'scorer_image_digest' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == ("stage1", None, NEW_SCORER_DIGEST)
            cursor.execute(
                "SELECT count(*),count(*) FILTER (WHERE status='accepted'),"
                "count(*) FILTER (WHERE terminal_cause='judge_error'),"
                "count(*) FILTER (WHERE terminal_cause='stage_closed') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (ARCHIVE,),
            )
            assert cursor.fetchone() == (49, 37, 2, 10)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='execute' AND status='accepted'",
                (ROUND,),
            )
            assert cursor.fetchone() == (98,)
            assert scoring_after.replace(
                rerun332._template_block("replacement"),
                rerun332._template_block("anchor"),
            ) == scoring_before
        connection.commit()


def test_cancelled339_template_is_inactive_and_has_no_release_binding():
    body = TEMPLATE.read_text()
    assert TEMPLATE.name not in CURRENT_SERVICE_MIGRATIONS
    assert "active.status IS DISTINCT FROM 'cancelled'" in body
    assert "score:rerun338" in body and "score:rerun339" in body
    assert "arena-2026-09-20-r339archive" in body
    assert CURRENT_SCORER_DIGEST in body
    assert "__NEW_SCORER_IMAGE_DIGEST__" in body
    assert "__RERUN_SCHEDULE_JSON__" in body
    assert "__TERMINAL_ACCEPTED_SCORE_COUNT__" in body
    assert "__TERMINAL_JUDGE_ERROR_COUNT__" in body
    assert "__TERMINAL_STAGE_CLOSED_COUNT__" in body
    assert "__TERMINAL_EXHAUSTED_SUBMISSION_SQL__" in body
