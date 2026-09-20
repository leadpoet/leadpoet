"""Sealed Sep20 scored/unpublished rejudge-only rerun335."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lab_arena.store import ArenaStoreError
from tests.lab_arena import sep20_authority_preserving_rejudge332_postgres_test as rerun332
from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS


ROUND = rerun332.ROUND
BASELINE = rerun332.BASELINE
PRIOR_ARCHIVES = (
    ROUND + "-r326archive",
    ROUND + "-r328archive",
    ROUND + "-r332archive",
)
ARCHIVE = ROUND + "-r335archive"
CURRENT_SCORER_DIGEST = (
    "sha256:cc2dd55dec0b94db8cd0e184d9bf67014e2e83def22d5c5b4e28213c46fe0cd8"
)
CURRENT_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + CURRENT_SCORER_DIGEST
)
NEW_SCORER_DIGEST = "sha256:" + "5b" * 32
NEW_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + NEW_SCORER_DIGEST
)
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/335-arena-2026-09-20-scored-authority-rejudge.sql.template"
)


@pytest.fixture()
def database():
    yield from rerun332.database.__wrapped__()


def _prepare_scored_rerun332(connection, harness, monkeypatch):
    rerun332._publish_rerun328(
        connection, harness, monkeypatch, zero_baseline=True
    )
    rerun332._inject_exhausted_provider_zero_and_failed_unknown(connection)
    schedule = rerun332._schedule()
    with connection.cursor() as cursor:
        monkeypatch.setattr(rerun332, "NEW_SCORER_DIGEST", CURRENT_SCORER_DIGEST)
        monkeypatch.setattr(rerun332, "NEW_SCORER_REFERENCE", CURRENT_SCORER_REFERENCE)
        rendered, _, _ = rerun332._render(cursor, schedule, harness.objects)
        cursor.execute(rendered)
    connection.commit()

    normal_breakdown = lifecycle._proof_breakdown

    def zero_baseline_breakdown(company, score):
        return normal_breakdown(company, 0 if float(score) > 0 else score)

    monkeypatch.setattr(lifecycle, "_proof_breakdown", zero_baseline_breakdown)
    rerun332._drive_rejudge_cycle(
        harness.service, harness.objects, harness.runner_keys[0]
    )
    monkeypatch.setattr(lifecycle, "_proof_breakdown", normal_breakdown)
    with connection.cursor() as cursor:
        assert rerun332._scalar(
            cursor,
            "SELECT status FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        ) == "scored"
        assert rerun332._json(
            cursor,
            "SELECT publication_doc FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        ) is None
        cursor.execute(
            "SELECT count(*),bool_and(status='accepted'),"
            "bool_and(assignment_id LIKE '%%:score:rerun332') "
            "FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='score' AND submission_id=%s",
            (ROUND, BASELINE),
        )
        assert cursor.fetchone() == (18, True, True)
        cursor.execute(
            "SELECT count(*),bool_and(per_icp_score=0) FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='execute' AND submission_id=%s "
            "AND status='accepted' AND terminal_cause='accepted'",
            (ROUND, BASELINE),
        )
        assert cursor.fetchone() == (18, True)
    connection.commit()
    return schedule


def _render335(cursor, schedule, objects, monkeypatch):
    monkeypatch.setattr(rerun332, "TEMPLATE", TEMPLATE)
    monkeypatch.setattr(rerun332, "RENDERED", TEMPLATE.with_suffix(""))
    monkeypatch.setattr(rerun332, "ARCHIVE", ARCHIVE)
    monkeypatch.setattr(rerun332, "PRIOR_ARCHIVES", PRIOR_ARCHIVES)
    monkeypatch.setattr(rerun332, "NEW_SCORER_DIGEST", NEW_SCORER_DIGEST)
    monkeypatch.setattr(rerun332, "NEW_SCORER_REFERENCE", NEW_SCORER_REFERENCE)
    return rerun332._render(cursor, schedule, objects)


def test_sep20_rerun335_preserves_scored_research_history_and_authority(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        schedule = _prepare_scored_rerun332(connection, harness, monkeypatch)
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
            prior_before = {
                round_id: rerun332._scope_snapshot(cursor, round_id)
                for round_id in PRIOR_ARCHIVES
            }
            old_cache_keys = set(
                rerun332._json(
                    cursor,
                    "SELECT jsonb_agg(judgment_cache_key) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND kind='score'",
                    (ROUND,),
                )
            )
            terminal_score_count = rerun332._scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            rendered, scoring_before, scoring_after = _render335(
                cursor, schedule, harness.objects, monkeypatch
            )
            values = rerun332._assert_exact_render_shape(rendered)
            assert values["__TERMINAL_STATUS__"] == "scored"
            assert values["__TERMINAL_CANCEL_REASON_SQL__"] == "NULL"
            cursor.execute(rendered)
            cursor.execute(rendered)

            assert rerun332._scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='score'",
                (ARCHIVE,),
            ) == terminal_score_count
            assert rerun332._scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute'",
                (ARCHIVE,),
            ) == "0"
            assert all(
                rerun332._scope_snapshot(cursor, round_id) == prior_before[round_id]
                for round_id in PRIOR_ARCHIVES
            )
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
            cursor.execute(
                "SELECT configuration_doc->>'scorer_image_digest',"
                "configuration_doc->>'scorer_image_reference',publication_doc,"
                "published_at FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                NEW_SCORER_DIGEST,
                NEW_SCORER_REFERENCE,
                None,
                None,
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
            assert rerun332._scalar(
                cursor,
                "SELECT status FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            ) == "scored"
            assert rerun332._json(
                cursor,
                "SELECT publication_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            ) is None
            assert rerun332._authority_snapshot(cursor) == authority_before
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),"
                "bool_and(status='accepted'),"
                "bool_and(assignment_id LIKE '%%:score:rerun335'),"
                "bool_and(judgment_scope_doc->>'scorer_image_digest'=%s) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (NEW_SCORER_DIGEST, ROUND),
            )
            assert cursor.fetchone() == (98, 98, True, True, True)
            new_cache_keys = set(
                rerun332._json(
                    cursor,
                    "SELECT jsonb_agg(judgment_cache_key) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND kind='score'",
                    (ROUND,),
                )
            )
            assert None not in new_cache_keys
            assert new_cache_keys.isdisjoint(old_cache_keys)
            assert rerun332._execution_snapshot(cursor) == execution_before
            cursor.execute(rendered)
            assert rerun332._authority_snapshot(cursor) == authority_before


def test_sep20_rerun335_refuses_terminal_drift_without_mutation(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        schedule = _prepare_scored_rerun332(connection, harness, monkeypatch)
        with connection.cursor() as cursor:
            rendered, _, _ = _render335(
                cursor, schedule, harness.objects, monkeypatch
            )
            cursor.execute("SAVEPOINT terminal_drift")
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET terminal_cause='worker_lost' "
                "WHERE round_id=%s AND kind='execute' AND status='failed' "
                "AND run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute' AND status='failed')",
                (ROUND, ROUND),
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


def test_sep20_rerun335_template_is_inactive_and_narrow():
    body = TEMPLATE.read_text()
    assert TEMPLATE.name not in CURRENT_SERVICE_MIGRATIONS
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert all(round_id in body for round_id in PRIOR_ARCHIVES)
    assert ARCHIVE in body and "score:rerun335" in body
    assert "assignment_id NOT LIKE '%:score:rerun332'" in body
    assert CURRENT_SCORER_DIGEST in body and CURRENT_SCORER_REFERENCE in body
    assert "__NEW_SCORER_IMAGE_DIGEST__" in body
    assert "__NEW_SCORER_IMAGE_REFERENCE__" in body
    assert "active.status IS DISTINCT FROM 'scored'" in body
    assert "active.publication_doc IS NOT NULL" in body
    assert "active.published_at IS NOT NULL" in body
    assert "lab_arena_000_sep20_rejudge332_publication_stop" in body
    assert "CREATE TRIGGER" not in body and "CREATE OR REPLACE FUNCTION" not in body
    assert "__TERMINAL_AUTHORITY_SHA256__" in body
    assert "authority_after IS DISTINCT FROM authority_before" in body
    assert "INSERT INTO public.lab_arena_runs" not in body
    assert "archived_execution_artifacts_sha256" in body
    assert '"openrouter":500' in body and '"openrouter":2000' not in body
    assert "INTERVAL '7 hours'" in body
    assert "INTERVAL '10 hours'" in body
    assert "INTERVAL '14 hours 1 second'" in body
    assert len(set(re.findall(r"__[A-Z0-9_]+__", body))) == 28
