"""Sealed Sep20 saved-output rerun338 from terminal rerun337 judgments."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from lab_arena.store import ArenaStoreError
from tests.lab_arena import sep20_authority_preserving_rejudge332_postgres_test as rerun332
from tests.lab_arena import sep20_scored_authority_rejudge337_postgres_test as rerun337
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS

ROUND = rerun332.ROUND
BASELINE = rerun332.BASELINE
ARCHIVE = ROUND + "-r338archive"
PRIOR_ARCHIVES = rerun337.PRIOR_ARCHIVES + (rerun337.ARCHIVE,)
CURRENT_SCORER_DIGEST = (
    "sha256:9b935f81fd80aaa8e1cd735d8f4d18607160d5adbd7085d82e7bad4ce8a4e09a"
)
CURRENT_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + CURRENT_SCORER_DIGEST
)
OLD_RERUN337_SCORER_DIGEST = rerun337.CURRENT_SCORER_DIGEST
OLD_RERUN337_SCORER_REFERENCE = rerun337.CURRENT_SCORER_REFERENCE
NEW_SCORER_DIGEST = "sha256:" + "6c" * 32
NEW_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + NEW_SCORER_DIGEST
)
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/338-arena-2026-09-20-scored-authority-rejudge.sql.template"
)
RENDERED = TEMPLATE.with_suffix("")
RENDERED_SHA256 = "a26b47663a0b71f23db591e592ceadb8861cf2909cb2cf3e2d3ea0d6282529e6"
FINAL_SCORER_DIGEST = (
    "sha256:acd64107ee3ed6d52d81708bf7cc951042eb0f02b6151d1197cd29fd6365dcfb"
)
FINAL_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + FINAL_SCORER_DIGEST
)


@pytest.fixture()
def database():
    yield from rerun332.database.__wrapped__()


def _prepare_scored337(connection, harness, monkeypatch):
    rerun337._prepare_scored335(connection, harness, monkeypatch)
    schedule = rerun332._schedule()
    with connection.cursor() as cursor:
        monkeypatch.setattr(rerun337, "NEW_SCORER_DIGEST", CURRENT_SCORER_DIGEST)
        monkeypatch.setattr(rerun337, "NEW_SCORER_REFERENCE", CURRENT_SCORER_REFERENCE)
        rendered337, _, _ = rerun337._render337(
            cursor, schedule, harness.objects, monkeypatch
        )
        cursor.execute(rendered337)
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
        # Production rerun337 retained one failed p8 attempt before its accepted
        # retry. Reproduce that exact terminal shape without adding provider work.
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
            "AND kind='score' AND submission_id=%s AND icp_position=8 "
            "AND status='accepted' AND terminal_cause='accepted'",
            (ROUND, BASELINE),
        )
        accepted_run_id = cursor.fetchone()[0]
        cursor.execute(
            "UPDATE public.lab_arena_runs SET attempt=2 WHERE run_id=%s",
            (accepted_run_id,),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record("
            "NULL::public.lab_arena_runs,to_jsonb(r)||jsonb_build_object("
            "'run_id',r.run_id||':failed-attempt-1','attempt',1,'status','failed',"
            "'terminal_cause','judge_error','result_doc',jsonb_build_object("
            "'terminal_status','judge_error','failure_diagnostic',jsonb_build_object("
            "'stage','scorer','error_class','judge_error','reason','unknown')),"
            "'claim_request_id',NULL,'claim_request_hash',NULL,'claim_response',NULL,"
            "'output_ref',NULL,'per_icp_score',NULL,'qualification_doc',NULL))).* "
            "FROM public.lab_arena_runs r WHERE r.run_id=%s",
            (accepted_run_id,),
        )
        cursor.execute("SET LOCAL session_replication_role=origin")
        cursor.execute(
            "SELECT count(*),count(*) FILTER (WHERE status='accepted'),"
            "count(*) FILTER (WHERE status='failed' AND terminal_cause='judge_error' "
            "AND submission_id=%s AND icp_position=8 AND attempt=1),"
            "bool_and(assignment_id LIKE '%%:score:rerun337') "
            "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
            (BASELINE, ROUND),
        )
        assert cursor.fetchone() == (99, 98, 1, True)
    connection.commit()


def _render338(cursor, schedule, objects, monkeypatch):
    monkeypatch.setattr(rerun332, "TEMPLATE", TEMPLATE)
    monkeypatch.setattr(rerun332, "RENDERED", RENDERED)
    monkeypatch.setattr(rerun332, "ARCHIVE", ARCHIVE)
    monkeypatch.setattr(rerun332, "PRIOR_ARCHIVES", PRIOR_ARCHIVES)
    monkeypatch.setattr(rerun332, "NEW_SCORER_DIGEST", NEW_SCORER_DIGEST)
    monkeypatch.setattr(rerun332, "NEW_SCORER_REFERENCE", NEW_SCORER_REFERENCE)
    return rerun332._render(cursor, schedule, objects)


def _assert_render_shape(rendered: str) -> dict[str, str]:
    template = TEMPLATE.read_text()
    marker_pattern = re.compile(r"__[A-Z0-9_]+__")
    parts = marker_pattern.split(template)
    markers = marker_pattern.findall(template)
    pattern = [re.escape(parts[0])]
    groups: dict[str, str] = {}
    for index, marker in enumerate(markers):
        group = groups.get(marker)
        if group is None:
            group = f"seal_{len(groups)}"
            groups[marker] = group
            pattern.append(f"(?P<{group}>.*?)")
        else:
            pattern.append(f"(?P={group})")
        pattern.append(re.escape(parts[index + 1]))
    match = re.fullmatch("".join(pattern), rendered, re.DOTALL)
    assert match is not None, "rendered recovery SQL changes non-placeholder bytes"
    values = {marker: match.group(group) for marker, group in groups.items()}
    assert all(value and "__" not in value for value in values.values())
    return values


def test_rerun338_verbatim_render_contract_preserves_execution_authority(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored337(connection, harness, monkeypatch)
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
                "SELECT jsonb_agg(to_jsonb(r)-'round_id'-'submission_id' "
                "ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            score_ledger = rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' "
                "ORDER BY entry_id) FROM public.lab_arena_ledger l "
                "WHERE round_id=%s AND EXISTS(SELECT 1 FROM public.lab_arena_runs r "
                "WHERE r.run_id=l.run_id AND r.round_id=%s AND r.kind='score')",
                (ROUND, ROUND),
            )
            prior_before = {
                round_id: rerun332._scope_snapshot(cursor, round_id)
                for round_id in PRIOR_ARCHIVES
            }
            schedule = rerun332._schedule()
            rendered, scoring_before, scoring_after = _render338(
                cursor, schedule, harness.objects, monkeypatch
            )
            fixture_values = _assert_render_shape(rendered)
            committed = RENDERED.read_text()
            assert hashlib.sha256(committed.encode()).hexdigest() == RENDERED_SHA256
            production_values = _assert_render_shape(committed)
            assert set(production_values) == set(fixture_values)
            assert production_values["__TERMINAL_STATUS__"] == "scored"
            assert production_values["__TERMINAL_RUN_COUNT__"] == "203"
            assert production_values["__TERMINAL_SCORE_RUN_COUNT__"] == "99"
            assert production_values["__TERMINAL_ACCEPTED_EXECUTION_COUNT__"] == "98"
            assert production_values["__NEW_SCORER_IMAGE_DIGEST__"] == (
                FINAL_SCORER_DIGEST
            )
            assert production_values["__NEW_SCORER_IMAGE_REFERENCE__"] == (
                FINAL_SCORER_REFERENCE
            )
            # The fixture swaps only declared seals in the committed render
            # contract. All executable SQL bytes outside those seals stay exact.
            fixture_from_committed_contract = TEMPLATE.read_text()
            for marker, value in fixture_values.items():
                fixture_from_committed_contract = (
                    fixture_from_committed_contract.replace(marker, value)
                )
            assert fixture_from_committed_contract == rendered
            cursor.execute(fixture_from_committed_contract)
            cursor.execute(fixture_from_committed_contract)

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
                "SELECT count(*),count(*) FILTER (WHERE status='accepted'),"
                "count(*) FILTER (WHERE status='failed' AND terminal_cause='judge_error' "
                "AND submission_id=%s AND icp_position=8 AND attempt=1),"
                "bool_and(assignment_id LIKE '%%:score:rerun337') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (BASELINE + ':r338archive', ARCHIVE),
            )
            assert cursor.fetchone() == (len(score_history), 98, 1, True)
            assert rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(r)-'round_id'-'submission_id' "
                "ORDER BY run_id) FROM public.lab_arena_runs r "
                "WHERE round_id=%s AND kind='score'",
                (ARCHIVE,),
            ) == score_history
            assert rerun332._json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' "
                "ORDER BY entry_id) FROM public.lab_arena_ledger l "
                "WHERE round_id=%s AND EXISTS(SELECT 1 FROM public.lab_arena_runs r "
                "WHERE r.run_id=l.run_id AND r.round_id=%s AND r.kind='score')",
                (ARCHIVE, ARCHIVE),
            ) == score_ledger
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
                "bool_and(assignment_id LIKE '%%:score:rerun338') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            assert cursor.fetchone() == (98, True, True)
            assert rerun332._authority_snapshot(cursor) == authority_before
            assert rerun332._execution_snapshot(cursor) == execution_before


def test_rerun338_template_and_numbered_render_contract_are_bounded():
    body = TEMPLATE.read_text()
    assert TEMPLATE.name not in CURRENT_SERVICE_MIGRATIONS
    assert RENDERED.name not in CURRENT_SERVICE_MIGRATIONS
    assert "assignment_id LIKE '%:score:rerun337'" in body
    assert "v_scored.icp_position::TEXT||':score:rerun338'" in body
    assert "arena-2026-09-20-r338archive" in body
    assert "arena-2026-09-20-r337archive" in body
    assert CURRENT_SCORER_DIGEST in body
    assert OLD_RERUN337_SCORER_DIGEST not in body
    assert "__NEW_SCORER_IMAGE_DIGEST__" in body
    assert "archived_execution_judgments" in body
    assert "__TERMINAL_COMPANY_JUDGMENTS_SHA256__" in body
    assert "__TERMINAL_JUDGMENT_CACHE_SHA256__" in body
    assert "Sep20 rerun338 USER trigger state differs" in body
    assert RENDERED.exists()
    committed = RENDERED.read_text()
    assert hashlib.sha256(committed.encode()).hexdigest() == RENDERED_SHA256
    values = _assert_render_shape(committed)
    assert values["__NEW_SCORER_IMAGE_DIGEST__"] == FINAL_SCORER_DIGEST
    assert values["__NEW_SCORER_IMAGE_REFERENCE__"] == FINAL_SCORER_REFERENCE
    assert values["__TERMINAL_RUN_COUNT__"] == "203"
    assert values["__TERMINAL_SCORE_RUN_COUNT__"] == "99"
    assert values["__TERMINAL_ACCEPTED_EXECUTION_COUNT__"] == "98"
    assert "2026-09-20T14:30:00Z" in values["__RERUN_SCHEDULE_JSON__"]
    assert re.search(r"__[A-Z0-9_]+__", committed) is None
    for migration in (TEMPLATE, RENDERED):
        prior_seals = re.findall(
            r"SELECT pg_catalog\.jsonb_build_object\(\n  'rounds'.*?"
            r"INTO prior_archive_(?:before|after);",
            migration.read_text(),
            re.DOTALL,
        )
        assert len(prior_seals) == 2
        for seal in prior_seals:
            assert "jsonb_agg" not in seal
            assert seal.count("'count',pg_catalog.count(*),'sha256'") == 4
            assert all(round_id in seal for round_id in PRIOR_ARCHIVES)
            assert len(re.findall(
                r"extensions\.digest\(\s*pg_catalog\.to_jsonb\(", seal
            )) == 4


def test_rerun338_rejects_terminal_drift_and_disabled_publication_stop(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored337(connection, harness, monkeypatch)
        with connection.cursor() as cursor:
            rendered, _, _ = _render338(
                cursor, rerun332._schedule(), harness.objects, monkeypatch
            )
            cursor.execute("SAVEPOINT terminal_drift")
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET assignment_id="
                "regexp_replace(assignment_id,'rerun337$','rerun336') "
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
                (OLD_RERUN337_SCORER_DIGEST, OLD_RERUN337_SCORER_REFERENCE, ROUND),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            cursor.execute("SAVEPOINT refused")
            with pytest.raises(psycopg2.Error, match="terminal preimage differs"):
                cursor.execute(rerun332._migration_body(rendered))
            cursor.execute("ROLLBACK TO SAVEPOINT refused")
            cursor.execute("ROLLBACK TO SAVEPOINT stale_rerun332_scorer")

            mutation = (
                "INTO prior_archive_before;\n"
                " ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;\n"
                " UPDATE public.lab_arena_rounds SET configuration_doc="
                "configuration_doc||'{\"prior_archive_mutation_probe\":true}'::jsonb "
                "WHERE round_id='arena-2026-09-20-r326archive';\n"
                " ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;"
            )
            mutated = rendered.replace(
                "INTO prior_archive_before;", mutation, 1
            )
            assert mutated != rendered
            before = rerun332._protected_database_snapshot(cursor)
            cursor.execute("SAVEPOINT refused")
            with pytest.raises(psycopg2.Error, match="rerun338 preservation differs"):
                cursor.execute(rerun332._migration_body(mutated))
            cursor.execute("ROLLBACK TO SAVEPOINT refused")
            assert rerun332._protected_database_snapshot(cursor) == before
