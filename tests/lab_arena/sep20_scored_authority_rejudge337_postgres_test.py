"""Sealed Sep20 saved-output rerun337 from accepted rerun335 judgments."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab_arena.store import ArenaStoreError
from tests.lab_arena import sep20_authority_preserving_rejudge332_postgres_test as rerun332
from tests.lab_arena import sep20_scored_authority_rejudge335_postgres_test as rerun335
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS

ROUND = rerun332.ROUND
BASELINE = rerun332.BASELINE
ARCHIVE = ROUND + "-r337archive"
PRIOR_ARCHIVES = rerun335.PRIOR_ARCHIVES + (rerun335.ARCHIVE,)
CURRENT_SCORER_DIGEST = (
    "sha256:c0342bcced7f7de2552427cbb6fcf62ab6f10f42a63c0004d1466727863343ec"
)
CURRENT_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + CURRENT_SCORER_DIGEST
)
OLD_RERUN332_SCORER_DIGEST = rerun335.CURRENT_SCORER_DIGEST
OLD_RERUN332_SCORER_REFERENCE = rerun335.CURRENT_SCORER_REFERENCE
NEW_SCORER_DIGEST = "sha256:" + "6c" * 32
NEW_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + NEW_SCORER_DIGEST
)
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/337-arena-2026-09-20-scored-authority-rejudge.sql.template"
)


@pytest.fixture()
def database():
    yield from rerun332.database.__wrapped__()


def _prepare_scored335(connection, harness, monkeypatch):
    schedule = rerun335._prepare_scored_rerun332(connection, harness, monkeypatch)
    with connection.cursor() as cursor:
        monkeypatch.setattr(rerun335, "NEW_SCORER_DIGEST", CURRENT_SCORER_DIGEST)
        monkeypatch.setattr(rerun335, "NEW_SCORER_REFERENCE", CURRENT_SCORER_REFERENCE)
        rendered335, _, _ = rerun335._render335(
            cursor, schedule, harness.objects, monkeypatch
        )
        cursor.execute(rendered335)
    connection.commit()
    rerun332._drive_rejudge_cycle(
        harness.service, harness.objects, harness.runner_keys[0]
    )
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT status,publication_doc,published_at FROM public.lab_arena_rounds "
            "WHERE round_id=%s",
            (ROUND,),
        )
        assert cursor.fetchone() == ("scored", None, None)
        cursor.execute(
            "SELECT count(*),bool_and(status='accepted'),"
            "bool_and(terminal_cause='accepted'),"
            "bool_and(assignment_id LIKE '%%:score:rerun335') "
            "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
            (ROUND,),
        )
        assert cursor.fetchone() == (98, True, True, True)
    connection.commit()


def _render337(cursor, schedule, objects, monkeypatch):
    monkeypatch.setattr(rerun332, "TEMPLATE", TEMPLATE)
    monkeypatch.setattr(rerun332, "RENDERED", TEMPLATE.with_suffix(""))
    monkeypatch.setattr(rerun332, "ARCHIVE", ARCHIVE)
    monkeypatch.setattr(rerun332, "PRIOR_ARCHIVES", PRIOR_ARCHIVES)
    monkeypatch.setattr(rerun332, "NEW_SCORER_DIGEST", NEW_SCORER_DIGEST)
    monkeypatch.setattr(rerun332, "NEW_SCORER_REFERENCE", NEW_SCORER_REFERENCE)
    return rerun332._render(cursor, schedule, objects)


def test_rerun337_archives_335_judges_and_preserves_execution_authority(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored335(connection, harness, monkeypatch)
        with connection.cursor() as cursor:
            authority_before = rerun332._authority_snapshot(cursor)
            execution_before = rerun332._execution_snapshot(cursor)
            judgments_before = rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) "
                "FROM public.lab_arena_company_judgments j",
            )
            cache_before = rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(c) ORDER BY cache_key) "
                "FROM public.lab_arena_judgment_cache c",
            )
            score_history = rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            prior_before = {
                round_id: rerun332._scope_snapshot(cursor, round_id)
                for round_id in PRIOR_ARCHIVES
            }
            schedule = rerun332._schedule()
            rendered, scoring_before, scoring_after = _render337(
                cursor, schedule, harness.objects, monkeypatch
            )
            values = rerun332._assert_exact_render_shape(rendered)
            assert values["__TERMINAL_STATUS__"] == "scored"
            cursor.execute(rendered)
            cursor.execute(rendered)

            assert rerun332._authority_snapshot(cursor) == authority_before
            assert rerun332._execution_snapshot(cursor) == execution_before
            assert rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) "
                "FROM public.lab_arena_company_judgments j",
            ) == judgments_before
            assert rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(c) ORDER BY cache_key) "
                "FROM public.lab_arena_judgment_cache c",
            ) == cache_before
            assert all(
                rerun332._scope_snapshot(cursor, round_id) == prior_before[round_id]
                for round_id in PRIOR_ARCHIVES
            )
            cursor.execute(
                "SELECT count(*),bool_and(status='accepted'),"
                "bool_and(assignment_id LIKE '%%:score:rerun335') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (ARCHIVE,),
            )
            assert cursor.fetchone() == (len(score_history), True, True)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute'",
                (ARCHIVE,),
            )
            assert cursor.fetchone() == (0,)
            cursor.execute(
                "SELECT jsonb_array_length(configuration_doc->"
                "'archived_execution_judgments') FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ARCHIVE,),
            )
            assert cursor.fetchone() == (len(execution_before["runs"]),)
            cursor.execute(
                "SELECT status,publication_doc,published_at,"
                "configuration_doc->>'scorer_image_digest' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                "stage1", None, None, NEW_SCORER_DIGEST
            )
            assert scoring_after.replace(
                rerun332._template_block("replacement"),
                rerun332._template_block("anchor"),
            ) == scoring_before
        connection.commit()

        rerun332._drive_rejudge_cycle(
            harness.service, harness.objects, harness.runner_keys[0]
        )
        with pytest.raises(
            ArenaStoreError,
            match="Sep20 rejudge332 publication requires sealed review release",
        ):
            harness.service.publish(ROUND)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),bool_and(status='accepted'),"
                "bool_and(assignment_id LIKE '%%:score:rerun337') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            assert cursor.fetchone() == (98, True, True)
            assert rerun332._authority_snapshot(cursor) == authority_before
            assert rerun332._execution_snapshot(cursor) == execution_before


def test_rerun337_template_is_inactive_and_namespace_bounded():
    body = TEMPLATE.read_text()
    assert TEMPLATE.name not in CURRENT_SERVICE_MIGRATIONS
    assert "assignment_id NOT LIKE '%:score:rerun335'" in body
    assert "v_scored.icp_position::TEXT||':score:rerun337'" in body
    assert "arena-2026-09-20-r337archive" in body
    assert "arena-2026-09-20-r335archive" in body
    assert CURRENT_SCORER_DIGEST in body
    assert OLD_RERUN332_SCORER_DIGEST not in body
    assert "__NEW_SCORER_IMAGE_DIGEST__" in body
    assert "archived_execution_judgments" in body
    assert "__TERMINAL_COMPANY_JUDGMENTS_SHA256__" in body
    assert "__TERMINAL_JUDGMENT_CACHE_SHA256__" in body
    assert "Sep20 rerun337 USER trigger state differs" in body


def test_rerun337_archives_nullable_empty_execution_judgment_metadata(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored335(connection, harness, monkeypatch)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='execute' AND status='accepted' ORDER BY run_id LIMIT 1",
                (ROUND,),
            )
            run_id = cursor.fetchone()[0]
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET per_icp_score=NULL,"
                "qualification_doc=NULL WHERE run_id=%s",
                (run_id,),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            rendered, _, _ = _render337(
                cursor, rerun332._schedule(), harness.objects, monkeypatch
            )
            cursor.execute(rendered)
            cursor.execute(
                "SELECT item->'per_icp_score',item->'qualification_doc' "
                "FROM public.lab_arena_rounds archived "
                "CROSS JOIN LATERAL jsonb_array_elements("
                "archived.configuration_doc->'archived_execution_judgments') item "
                "WHERE archived.round_id=%s AND item->>'run_id'=%s",
                (ARCHIVE, run_id),
            )
            assert cursor.fetchone() == (None, None)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='execute' AND status='accepted' AND terminal_cause='accepted'",
                (ROUND,),
            )
            assert cursor.fetchone() == (98,)


def test_rerun337_rejects_terminal_drift_and_disabled_publication_stop(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored335(connection, harness, monkeypatch)
        with connection.cursor() as cursor:
            rendered, _, _ = _render337(
                cursor, rerun332._schedule(), harness.objects, monkeypatch
            )
            cursor.execute("SAVEPOINT terminal_drift")
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET assignment_id="
                "regexp_replace(assignment_id,'rerun335$','rerun334') "
                "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='score')",
                (ROUND,),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            before = rerun332._protected_database_snapshot(cursor)
            cursor.execute("SAVEPOINT refused")
            with pytest.raises(
                psycopg2.Error, match=r"terminal (?:preimage|history) differs"
            ):
                cursor.execute(rerun332._migration_body(rendered))
            cursor.execute("ROLLBACK TO SAVEPOINT refused")
            assert rerun332._protected_database_snapshot(cursor) == before
            cursor.execute("ROLLBACK TO SAVEPOINT terminal_drift")

            cursor.execute("SAVEPOINT missing_stop")
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_000_sep20_rejudge332_publication_stop"
            )
            before = rerun332._protected_database_snapshot(cursor)
            cursor.execute("SAVEPOINT refused")
            with pytest.raises(psycopg2.Error, match="publication stop differs"):
                cursor.execute(rerun332._migration_body(rendered))
            cursor.execute("ROLLBACK TO SAVEPOINT refused")
            assert rerun332._protected_database_snapshot(cursor) == before
            cursor.execute("ROLLBACK TO SAVEPOINT missing_stop")

            cursor.execute("SAVEPOINT disabled_user_trigger")
            cursor.execute(
                "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER "
                "lab_arena_runs_terminal"
            )
            before = rerun332._protected_database_snapshot(cursor)
            cursor.execute("SAVEPOINT refused")
            with pytest.raises(psycopg2.Error, match="USER trigger state differs"):
                cursor.execute(rerun332._migration_body(rendered))
            cursor.execute("ROLLBACK TO SAVEPOINT refused")
            assert rerun332._protected_database_snapshot(cursor) == before
            cursor.execute(
                "SELECT tgenabled FROM pg_trigger WHERE tgrelid="
                "'public.lab_arena_runs'::regclass "
                "AND tgname='lab_arena_runs_terminal'"
            )
            assert cursor.fetchone() == ("D",)
            cursor.execute("ROLLBACK TO SAVEPOINT disabled_user_trigger")

            cursor.execute("SAVEPOINT stale_rerun332_scorer")
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc="
                "jsonb_set(jsonb_set(configuration_doc,'{scorer_image_digest}',"
                "to_jsonb(%s::text)),'{scorer_image_reference}',to_jsonb(%s::text)) "
                "WHERE round_id=%s",
                (OLD_RERUN332_SCORER_DIGEST, OLD_RERUN332_SCORER_REFERENCE, ROUND),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            cursor.execute("SAVEPOINT refused")
            with pytest.raises(psycopg2.Error, match="terminal preimage differs"):
                cursor.execute(rerun332._migration_body(rendered))
            cursor.execute("ROLLBACK TO SAVEPOINT refused")
            cursor.execute("ROLLBACK TO SAVEPOINT stale_rerun332_scorer")
