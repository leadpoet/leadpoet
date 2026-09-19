"""Disposable PostgreSQL proof for the sealed Sep19 rerun313 template."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena import sep19_terminal309_newjudge_rerun310_postgres_test as rerun310
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "scripts/313-arena-2026-09-19-terminal310-fixed-baseline-rerun.sql.template"
ROUND = rerun310.ROUND
BASELINE = rerun310.BASELINE
NEW_SOURCE_REF = "arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun313.tar.gz"
NEW_SOURCE_SIZE = 700_313
NEW_SOURCE_SHA = "1" * 64
NEW_SOURCE_COMMIT = "2" * 40
NEW_SCORER_DIGEST = "sha256:" + "4" * 64
NEW_SCORER_REFERENCE = "registry.test/judge@" + NEW_SCORER_DIGEST


def _test_migrations() -> tuple[str, ...]:
    return rerun310._test_migrations() + (
        "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
        "312-lab-arena-temporary-hold-admission.sql",
    )


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(_test_migrations())


def _schedule() -> dict:
    start = datetime.now(timezone.utc) + timedelta(seconds=5)
    stamp = lambda value: value.isoformat().replace("+00:00", "Z")
    return {
        "submission_open": "2026-09-18T00:00:00Z",
        "submission_cutoff": "2026-09-19T00:00:00Z",
        "benchmark_deadline": stamp(start),
        "stage_1_start": stamp(start + timedelta(seconds=1)),
        "stage_1_close": stamp(start + timedelta(hours=3, seconds=1)),
        "stage_1_scoring_close": stamp(start + timedelta(hours=6)),
        "stage_2_start": stamp(start + timedelta(hours=6, seconds=1)),
        "stage_2_close": stamp(start + timedelta(hours=9, seconds=1)),
        "final_scoring_close": stamp(start + timedelta(hours=12)),
        "publication_deadline": stamp(start + timedelta(hours=12, seconds=1)),
    }


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(cursor, query: str, parameters=()) -> str:
    cursor.execute(query, parameters)
    return cursor.fetchone()[0]


def _scorer_patch(definition: str) -> str:
    old_guard = """  IF p_round_id='arena-2026-09-19'
     AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id='arena-2026-09-19-r310archive')
     AND public.lab_arena_sep19_rerun310_active_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_sep19_rerun310_frozen_state_invalid'
      USING ERRCODE='22023';
  END IF;
  FOR v_item IN"""
    new_guard = """  IF p_round_id='arena-2026-09-19'
     AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id='arena-2026-09-19-r313archive')
     AND public.lab_arena_sep19_rerun313_active_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_sep19_rerun313_frozen_state_invalid'
      USING ERRCODE='22023';
  END IF;
  FOR v_item IN"""
    old_assignment = """    IF p_round_id='arena-2026-09-19'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-19-r310archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun310';
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;"""
    new_assignment = """    IF p_round_id='arena-2026-09-19'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-19-r313archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun313';
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;"""
    assert definition.count(old_guard) == 1
    assert definition.count(old_assignment) == 1
    return definition.replace(old_guard, new_guard).replace(old_assignment, new_assignment)


def _seed_terminal_rerun310(conn) -> None:
    rerun310._seed_published_recovery309(conn)
    schedule = rerun310._schedule()
    with conn.cursor() as cursor:
        cursor.execute(rerun310._render(cursor, schedule))
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            """
            UPDATE public.lab_arena_runs
            SET status='accepted',terminal_cause='accepted',runner_hotkey=miner_hotkey,
                output_ref='arena/test/rerun310/'||run_id||'.json',
                result_doc='{}'::jsonb,participation_accepted_at=clock_timestamp(),
                per_icp_score=1::double precision,
                qualification_doc=jsonb_build_object('fixture','terminal310')
            WHERE round_id=%s AND kind='execute' AND submission_id=%s
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 20
        cursor.execute(
            """
            INSERT INTO public.lab_arena_runs(
              run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
              icp_position,attempt,kind,status,stage_generation,runner_hotkey,
              terminal_cause,output_ref,scored_run_id,judgment_cache_key,
              judgment_input_hash,judgment_scope_doc
            )
            SELECT execute.round_id||':'||execute.submission_id||':'||execute.stage::text||':'||
                     execute.icp_position::text||':score:rerun310:1',
                   execute.round_id||':'||execute.submission_id||':'||execute.stage::text||':'||
                     execute.icp_position::text||':score:rerun310',
                   execute.round_id,execute.submission_id,execute.miner_hotkey,
                   execute.stage,execute.icp_position,1,
                   'score','accepted',execute.stage_generation,execute.miner_hotkey,'accepted',
                   'arena/test/rerun310/judgment/'||execute.run_id||'.json',execute.run_id,
                   'sha256:'||repeat('6',64),'sha256:'||repeat('7',64),
                   jsonb_build_object(
                     'scorer_image_digest',round.configuration_doc->>'scorer_image_digest',
                     'scorer_image_reference',round.configuration_doc->>'scorer_image_reference')
            FROM public.lab_arena_runs execute
            JOIN public.lab_arena_rounds round USING(round_id)
            WHERE execute.round_id=%s AND execute.kind='execute'
              AND execute.status='accepted'
            """,
            (ROUND,),
        )
        assert cursor.rowcount == 100
        cursor.execute(
            """
            UPDATE public.lab_arena_rounds
            SET status='published',published_at=clock_timestamp(),
                publication_doc=jsonb_build_object(
                  'schema_version','leadpoet.lab_arena.publication.v1',
                  'king_decision',jsonb_build_object('outcome','no_king')),
                king_outcome='no_king',cancel_reason=NULL
            WHERE round_id=%s
            """,
            (ROUND,),
        )
        cursor.execute("SET session_replication_role=origin")
        cursor.execute("SELECT public.lab_arena_sep19_rerun310_active_valid_v1()")
        assert cursor.fetchone()[0] is True
    conn.commit()


def _render(cursor, schedule: dict) -> tuple[str, str, str]:
    cursor.execute(
        "SELECT pg_get_functiondef("
        "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
    )
    old_definition = cursor.fetchone()[0]
    new_definition = _scorer_patch(old_definition)
    values = {
        "__NEW_SOURCE_REF__": NEW_SOURCE_REF,
        "__NEW_SOURCE_SIZE_BYTES__": str(NEW_SOURCE_SIZE),
        "__NEW_SOURCE_SHA256__": NEW_SOURCE_SHA,
        "__NEW_SOURCE_COMMIT__": NEW_SOURCE_COMMIT,
        "__NEW_SCORER_DIGEST__": NEW_SCORER_DIGEST,
        "__NEW_SCORER_REFERENCE__": NEW_SCORER_REFERENCE,
        "__RERUN_SCHEDULE_JSON__": _compact(schedule),
        "__SCORING_DEFINITION_SHA256__": hashlib.sha256(old_definition.encode()).hexdigest(),
        "__PATCHED_SCORING_DEFINITION_SHA256__": hashlib.sha256(
            new_definition.encode()
        ).hexdigest(),
        "__TERMINAL_EXECUTE_RUN_COUNT__": "100",
        "__TERMINAL_BASELINE_RUN_COUNT__": "20",
        "__TERMINAL_SCORE_RUN_COUNT__": "100",
        "__TERMINAL_ROUND_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') "
            "FROM public.lab_arena_rounds r WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_BASELINE_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(to_jsonb(s)::text,'sha256'),'hex') "
            "FROM public.lab_arena_submissions s WHERE round_id=%s AND submission_id=%s",
            (ROUND, BASELINE),
        ),
        "__TERMINAL_MINER_SUBMISSIONS_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode(extensions.digest("
            "to_jsonb(s)::text,'sha256'),'hex'),'' ORDER BY submission_id),''),'sha256'),'hex') "
            "FROM public.lab_arena_submissions s WHERE round_id=%s AND submission_id<>%s",
            (ROUND, BASELINE),
        ),
        "__TERMINAL_RUNS_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode(extensions.digest("
            "to_jsonb(r)::text,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex') "
            "FROM public.lab_arena_runs r WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_LEDGER_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode(extensions.digest("
            "to_jsonb(l)::text,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') "
            "FROM public.lab_arena_ledger l WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_BANK_STATE_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(jsonb_build_object("
            "'confirmation_bank_ref',confirmation_bank_ref,"
            "'confirmation_bank_hash',confirmation_bank_hash,"
            "'confirmation_cohort',confirmation_cohort)::text,'sha256'),'hex') "
            "FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_REWARD_AUTHORITY_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(jsonb_build_object("
            "'rewards_enabled',rewards_enabled,"
            "'effective_reward_epoch',effective_reward_epoch,"
            "'reward_basis_hash',reward_basis_hash,'reward_basis_doc',reward_basis_doc,"
            "'signing_key_doc',signing_key_doc,'reward_activated_at',reward_activated_at,"
            "'king_outcome',king_outcome,'king_hotkey',king_hotkey,"
            "'king_start_epoch',king_start_epoch,'promotion_required',promotion_required,"
            "'promotion_doc',promotion_doc,'baseline_promoted_at',baseline_promoted_at)"
            "::text,'sha256'),'hex') FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        ),
    }
    sql = TEMPLATE.read_text()
    assert set(re.findall(r"__[A-Z0-9_]+__", sql)) == set(values)
    for marker, value in values.items():
        sql = sql.replace(marker, value)
    assert re.search(r"__[A-Z0-9_]+__", sql) is None
    return sql, values["__SCORING_DEFINITION_SHA256__"], values[
        "__PATCHED_SCORING_DEFINITION_SHA256__"
    ]


def _function_seals(cursor) -> dict[str, str]:
    signatures = (
        "public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)",
        "public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)",
        "public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)",
        "public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)",
    )
    return {
        signature: _sha(
            cursor,
            "SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),'sha256'),'hex')",
            (signature,),
        )
        for signature in signatures
    }


def test_terminal310_rerun_executes_scores_and_publishes_with_preservation(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_terminal_rerun310(conn)
        schedule = _schedule()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s "
                "AND submission_id<>%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            miner_runs_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r313archive"),
            )
            prior_ledger_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r "
                "WHERE round_id='arena-2026-09-19-r310archive'"
            )
            rerun310_archive_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT rewards_enabled,effective_reward_epoch,reward_basis_hash,"
                "reward_basis_doc,signing_key_doc,reward_activated_at,king_outcome,"
                "king_hotkey,king_start_epoch,promotion_required,promotion_doc,"
                "baseline_promoted_at FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            reward_state_before = cursor.fetchone()
            behavior_seals_before = _function_seals(cursor)
            rendered, old_scorer_hash, new_scorer_hash = _render(cursor, schedule)

        attacked = rendered.replace(old_scorer_hash, "0" * 64, 1)
        with pytest.raises(psycopg2.Error, match="Sep19 rerun313 scorer definition differs"):
            with conn.cursor() as cursor:
                cursor.execute(attacked)
        conn.rollback()

        with conn.cursor() as cursor:
            terminal_round_hash = _sha(
                cursor,
                "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') "
                "FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND,),
            )
        attacked = rendered.replace(terminal_round_hash, "f" * 64)
        with pytest.raises(psycopg2.Error, match="Sep19 rerun313 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(attacked)
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute(rendered)
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE status='pending'),"
                "bool_and(assignment_id LIKE '%%:rerun313') "
                "FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20, True)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
            )
            definition = cursor.fetchone()[0]
            assert hashlib.sha256(definition.encode()).hexdigest() == new_scorer_hash
            assert "lab_arena_sep19_rerun310_active_valid_v1()" not in definition
            assert ":score:rerun310" not in definition
            assert "lab_arena_sep19_rerun313_active_valid_v1()" in definition
            assert ":score:rerun313" in definition
            cursor.execute(
                "SELECT tgname FROM pg_trigger WHERE tgrelid='public.lab_arena_runs'::regclass "
                "AND NOT tgisinternal AND tgname LIKE 'lab_arena_sep19_rerun%%score_namespace_guard' "
                "ORDER BY tgname"
            )
            assert cursor.fetchall() == [
                ("lab_arena_sep19_rerun313_score_namespace_guard",)
            ]
            assert _function_seals(cursor) == behavior_seals_before
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r "
                "WHERE round_id='arena-2026-09-19-r310archive'"
            )
            assert cursor.fetchone()[0] == rerun310_archive_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r313archive"),
            )
            assert cursor.fetchone()[0] == prior_ledger_before
            cursor.execute(rendered)
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep19_rerun313_v1("
                "%s,%s,%s,%s::jsonb,%s,%s)",
                (
                    NEW_SOURCE_SIZE,
                    NEW_SOURCE_SHA,
                    NEW_SOURCE_COMMIT,
                    json.dumps(schedule),
                    NEW_SCORER_DIGEST,
                    NEW_SCORER_REFERENCE,
                ),
            )
            assert cursor.fetchone()[0]["status"] == "existing"
        conn.commit()

        monkeypatch.setattr(lifecycle, "ROUND", ROUND)
        monkeypatch.setattr(lifecycle, "BASELINE", BASELINE)
        monkeypatch.setattr(lifecycle, "_proof_execution", rerun310._proof_execution_v2)
        proof_breakdown = lifecycle._proof_breakdown
        monkeypatch.setattr(
            lifecycle,
            "_proof_breakdown",
            lambda company, score: proof_breakdown(company, 40 if score else 39),
        )
        harness = lifecycle.Harness(
            lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha", "beta"]
        )
        harness.round_id = ROUND
        harness.objects.put(
            f"arena/{ROUND}/benchmark.json", rerun310.BENCHMARK_BYTES
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT run_id,output_ref,icp_position FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute' AND submission_id<>%s",
                (ROUND, BASELINE),
            )
            for index, (run_id, output_ref, position) in enumerate(cursor.fetchall(), 1):
                rerun310._proof_execution_v2(
                    harness.objects,
                    rerun310.BENCHMARK["icps"][position],
                    index,
                    position,
                    run_id,
                )
                harness.objects.put(
                    output_ref, harness.objects.get(f"arena/output/{run_id}.json")
                )
        conn.commit()

        lifecycle._drive_cycle(
            harness.service,
            harness.objects,
            rerun310.BENCHMARK["icps"],
            harness.runner_keys[1],
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT count(DISTINCT assignment_id),"
                "count(DISTINCT assignment_id) FILTER(WHERE status='accepted') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' "
                "AND submission_id=%s",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20)
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE assignment_id LIKE '%%:score:rerun313'),"
                "bool_and(judgment_scope_doc->>'scorer_image_digest'=%s),"
                "bool_and(judgment_scope_doc->>'scorer_image_reference'=%s) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' "
                "AND status='accepted'",
                (NEW_SCORER_DIGEST, NEW_SCORER_REFERENCE, ROUND),
            )
            assert cursor.fetchone() == (100, 100, True, True)
        conn.commit()

        assert harness.service.publish(ROUND)["status"] == "ok"
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT rewards_enabled,effective_reward_epoch,reward_basis_hash,"
                "reward_basis_doc,signing_key_doc,reward_activated_at,king_outcome,"
                "king_hotkey,king_start_epoch,promotion_required,promotion_doc,"
                "baseline_promoted_at FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == reward_state_before
            cursor.execute(
                "SELECT status,jsonb_array_length(publication_doc->'final_ranking') "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == ("published", 5)
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r "
                "WHERE round_id='arena-2026-09-19-r310archive'"
            )
            assert cursor.fetchone()[0] == rerun310_archive_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r313archive"),
            )
            assert cursor.fetchone()[0] == prior_ledger_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s "
                "AND submission_id<>%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            miner_runs_after = cursor.fetchone()[0]
            for before, after in zip(miner_runs_before, miner_runs_after, strict=True):
                for field in ("per_icp_score", "qualification_doc", "updated_at"):
                    before.pop(field, None)
                    after.pop(field, None)
            assert miner_runs_after == miner_runs_before
    finally:
        conn.close()


def test_template_is_sealed_terminal310_only():
    body = TEMPLATE.read_text()
    assert "after the Sep19 rerun310 round is terminal" in body
    assert "assignment_id LIKE '%:rerun310'" in body
    assert "assignment_id NOT LIKE '%:score:rerun310'" in body
    assert "assignment_id LIKE '%:recovery309'" not in body
    assert body.count("CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_v2") == 0
    assert "old_guard TEXT:=$guard310$" in body
    assert "old_assignment TEXT:=$assignment310$" in body
    assert "lab_arena_sep19_rerun310_active_valid_v1()" in body
    assert "lab_arena_sep19_rerun313_active_valid_v1()" in body
    assert "DROP TRIGGER IF EXISTS lab_arena_sep19_rerun310_score_namespace_guard" in body
    assert "__SCORING_DEFINITION_SHA256__" in body
    assert "__PATCHED_SCORING_DEFINITION_SHA256__" in body
    assert "__NEW_SOURCE_COMMIT__" in body
    assert "__NEW_SCORER_DIGEST__" in body
    assert "__RERUN_SCHEDULE_JSON__" in body
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert "lab_arena_accepted_weight_states" in body
    assert "company_quality_policy" in body
