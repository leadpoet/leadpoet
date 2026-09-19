"""Disposable PostgreSQL proof for the sealed Sep19 rerun316 template."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena import (
    sep19_terminal313_fixed_baseline_rerun315_postgres_test as prior,
)
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "scripts/316-arena-2026-09-19-terminal315-fixed-baseline-rerun.sql.template"
rerun310 = prior.prior.rerun310
ROUND = prior.ROUND
BASELINE = prior.BASELINE
# Recovery316 binds a new source and scorer at render time while creating a
# fresh assignment namespace. Fixtures reuse immutable local bytes only.
NEW_SOURCE_REF = prior.NEW_SOURCE_REF
NEW_SOURCE_SIZE = prior.NEW_SOURCE_SIZE
NEW_SOURCE_SHA = prior.NEW_SOURCE_SHA
NEW_SOURCE_COMMIT = prior.NEW_SOURCE_COMMIT
NEW_SCORER_DIGEST = prior.NEW_SCORER_DIGEST
NEW_SCORER_REFERENCE = prior.NEW_SCORER_REFERENCE


def _test_migrations() -> tuple[str, ...]:
    return prior._test_migrations()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(_test_migrations())


@pytest.fixture(scope="module")
def terminal_failure_database():
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


def _template_block(name: str) -> str:
    body = TEMPLATE.read_text()
    matches = re.findall(rf"\${re.escape(name)}\$(.*?)\${re.escape(name)}\$", body, re.DOTALL)
    assert len(matches) == 1
    return matches[0]


def _scorer_patch(definition: str) -> str:
    old_guard = _template_block("guard315")
    new_guard = _template_block("new_guard")
    old_assignment = _template_block("assignment315")
    new_assignment = _template_block("new_assignment")
    assert definition.count(old_guard) == 1
    assert definition.count(old_assignment) == 1
    return definition.replace(old_guard, new_guard).replace(old_assignment, new_assignment)


def _seed_terminal_rerun315(conn) -> None:
    prior._seed_terminal_rerun313_failure(conn)
    schedule = prior._schedule()
    with conn.cursor() as cursor:
        rendered, _, _ = prior._render(cursor, schedule)
        cursor.execute(rendered)
    conn.commit()

    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            """
            UPDATE public.lab_arena_runs
            SET status=CASE WHEN icp_position=0 THEN 'accepted' ELSE 'failed' END,
                terminal_cause=CASE WHEN icp_position=0 THEN 'accepted' ELSE 'model_error' END,
                runner_hotkey=miner_hotkey,
                result_doc=jsonb_build_object('terminal_status',
                  CASE WHEN icp_position=0 THEN 'accepted' ELSE 'model_error' END),
                output_ref=CASE WHEN icp_position=0
                  THEN 'arena/output/rerun315-accepted.json' ELSE NULL END
            WHERE round_id=%s AND kind='execute' AND submission_id=%s
              AND assignment_id LIKE '%%:rerun315' AND attempt=1
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 20
        cursor.execute(
            """
            INSERT INTO public.lab_arena_runs(
              run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
              icp_position,attempt,kind,status,stage_generation,runner_hotkey,
              previous_runner_hotkey,terminal_cause,result_doc)
            SELECT assignment_id||':2',assignment_id,round_id,submission_id,
                   miner_hotkey,stage,icp_position,2,'execute',
                   'failed',
                   stage_generation,miner_hotkey,miner_hotkey,
                   CASE WHEN icp_position=19 THEN 'provider_error' ELSE 'model_error' END,
                   jsonb_build_object('terminal_status',
                     CASE WHEN icp_position=19 THEN 'provider_error' ELSE 'model_error' END)
            FROM public.lab_arena_runs
            WHERE round_id=%s AND kind='execute' AND submission_id=%s
              AND assignment_id LIKE '%%:rerun315' AND attempt=1 AND icp_position>0
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 19
        # This is sealed historical evidence. It represents two completed paid
        # calls and proves that recovery moves every ledger entry unchanged.
        cursor.execute(
            """
            WITH selected AS (
              SELECT run_id,miner_hotkey,stage,
                     row_number() OVER (ORDER BY status DESC,run_id) AS n
              FROM public.lab_arena_runs
              WHERE round_id=%s AND submission_id=%s AND kind='execute'
              ORDER BY status DESC,run_id LIMIT 2
            ), calls AS (
              SELECT selected.*,
                     'sha256:'||encode(extensions.digest(
                       ('rerun315-paid-'||n::text)::bytea,'sha256'),'hex') AS identity,
                     CASE WHEN n=1 THEN 15165 ELSE 34000 END AS actual
              FROM selected
            ), entries AS (
              SELECT calls.*,entry_kind,ordinality
              FROM calls CROSS JOIN unnest(ARRAY['reservation','dispatch','settlement'])
                WITH ORDINALITY AS kinds(entry_kind,ordinality)
            )
            INSERT INTO public.lab_arena_ledger(
              entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              call_identity,provider,operation_id,funding_source,amount_microusd,
              entry_doc,terminal_response)
            SELECT entry_kind,miner_hotkey,%s,%s,run_id,stage,identity,
                   CASE WHEN n=1 THEN 'openrouter' ELSE 'deepline' END,
                   CASE WHEN n=1 THEN 'openrouter.responses' ELSE 'deepline.execute' END,
                   'host',CASE WHEN entry_kind='reservation' THEN 4000000
                               WHEN entry_kind='settlement' THEN actual ELSE 0 END,
                   jsonb_build_object('sealed_fixture',TRUE,'sequence',ordinality),
                   CASE WHEN entry_kind='settlement'
                     THEN jsonb_build_object('call_succeeded',TRUE)
                     ELSE NULL END
            FROM entries ORDER BY n,ordinality
            """,
            (ROUND, BASELINE, ROUND, BASELINE),
        )
        assert cursor.rowcount == 6
        cursor.execute(
            """
            UPDATE public.lab_arena_rounds
            SET status='cancelled',published_at=NULL,publication_doc=NULL,
                cancel_reason='execution_incomplete:stage1:5',
                updated_at=clock_timestamp()
            WHERE round_id=%s
            """,
            (ROUND,),
        )
        assert cursor.rowcount == 1
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            """
            SELECT public.lab_arena_sep19_rerun315_active_valid_v1(),
                   count(*) FILTER(WHERE r.status='accepted'),
                   count(*) FILTER(WHERE r.status='failed'),
                   count(*) FILTER(WHERE r.terminal_cause='model_error'),
                   count(*) FILTER(WHERE r.terminal_cause='provider_error'),
                   count(DISTINCT r.assignment_id)
            FROM public.lab_arena_runs r
            WHERE r.round_id=%s AND r.kind='execute' AND r.submission_id=%s
            GROUP BY public.lab_arena_sep19_rerun315_active_valid_v1()
            """,
            (ROUND, BASELINE),
        )
        assert cursor.fetchone() == (True, 1, 38, 37, 1, 20)
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
            (ROUND,),
        )
        assert cursor.fetchone()[0] == 0
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s "
            "AND submission_id=%s",
            (ROUND, BASELINE),
        )
        assert cursor.fetchone()[0] == 6
    conn.commit()


def _render(cursor, schedule: dict) -> tuple[str, str, str]:
    cursor.execute(
        "SELECT pg_get_functiondef("
        "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
    )
    old_definition = cursor.fetchone()[0]
    new_definition = _scorer_patch(old_definition)
    cursor.execute(
        "SELECT status,cancel_reason FROM public.lab_arena_rounds WHERE round_id=%s",
        (ROUND,),
    )
    terminal_status, terminal_cancel_reason = cursor.fetchone()
    assert re.fullmatch(r"[a-z0-9_]+", terminal_status)
    assert re.fullmatch(r"[a-z0-9_:.-]+", terminal_cancel_reason)
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
        "__TERMINAL_STATUS__": terminal_status,
        "__TERMINAL_CANCEL_REASON__": terminal_cancel_reason,
        "__TERMINAL_EXECUTE_RUN_COUNT__": str(_sha(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='execute'",
            (ROUND,),
        )),
        "__TERMINAL_BASELINE_RUN_COUNT__": str(_sha(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='execute' AND submission_id=%s",
            (ROUND, BASELINE),
        )),
        "__TERMINAL_SCORE_RUN_COUNT__": str(_sha(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='score'",
            (ROUND,),
        )),
        "__TERMINAL_ACCEPTED_EXECUTE_RUN_COUNT__": str(_sha(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='execute' AND status='accepted' "
            "AND terminal_cause='accepted' AND output_ref IS NOT NULL",
            (ROUND,),
        )),
        "__TERMINAL_BASELINE_ACCEPTED_RUN_COUNT__": str(_sha(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='execute' AND submission_id=%s "
            "AND status='accepted'",
            (ROUND, BASELINE),
        )),
        "__TERMINAL_BASELINE_FAILED_RUN_COUNT__": str(_sha(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='execute' AND submission_id=%s "
            "AND status='failed'",
            (ROUND, BASELINE),
        )),
        "__TERMINAL_BASELINE_LEDGER_COUNT__": str(_sha(
            cursor,
            "SELECT count(*) FROM public.lab_arena_ledger "
            "WHERE round_id=%s AND submission_id=%s",
            (ROUND, BASELINE),
        )),
        "__TERMINAL_SCORE_LEDGER_COUNT__": str(_sha(
            cursor,
            "SELECT count(*) FROM public.lab_arena_ledger l WHERE round_id=%s "
            "AND EXISTS(SELECT 1 FROM public.lab_arena_runs r "
            "WHERE r.round_id=%s AND r.kind='score' AND r.run_id=l.run_id)",
            (ROUND, ROUND),
        )),
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


def _round_history(cursor, round_id: str):
    cursor.execute(
        "SELECT jsonb_build_object("
        "'round',(SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s),"
        "'submissions',(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
        " FROM public.lab_arena_submissions s WHERE round_id=%s),"
        "'runs',(SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) "
        " FROM public.lab_arena_runs r WHERE round_id=%s),"
        "'ledger',(SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
        " FROM public.lab_arena_ledger l WHERE round_id=%s))",
        (round_id, round_id, round_id, round_id),
    )
    return cursor.fetchone()[0]


def test_terminal315_rerun_executes_scores_and_publishes_with_preservation(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_terminal_rerun315(conn)
        schedule = _schedule()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT s.source_ref,s.source_size_bytes,"
                "s.submission_doc->>'source_sha256',s.submission_doc->>'source_commit',"
                "r.configuration_doc->>'scorer_image_digest',"
                "r.configuration_doc->>'scorer_image_reference' "
                "FROM public.lab_arena_submissions s "
                "JOIN public.lab_arena_rounds r USING(round_id) "
                "WHERE s.round_id=%s AND s.submission_id=%s",
                (ROUND, BASELINE),
            )
            source_and_scorer_before = cursor.fetchone()
            assert source_and_scorer_before == (
                NEW_SOURCE_REF,
                NEW_SOURCE_SIZE,
                NEW_SOURCE_SHA,
                NEW_SOURCE_COMMIT,
                NEW_SCORER_DIGEST,
                NEW_SCORER_REFERENCE,
            )
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
                (ROUND, ROUND + "-r316archive"),
            )
            prior_ledger_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r)-'round_id'-'submission_id' ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='execute' "
                "AND submission_id=%s",
                (ROUND, BASELINE),
            )
            terminal_baseline_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s AND submission_id=%s",
                (ROUND, BASELINE),
            )
            terminal_baseline_ledger_before = cursor.fetchone()[0]
            assert len(terminal_baseline_ledger_before) == 6
            cursor.execute(
                "SELECT jsonb_agg(jsonb_build_object("
                "'run_id',run_id,'per_icp_score',per_icp_score,"
                "'qualification_doc',qualification_doc,'updated_at',updated_at) "
                "ORDER BY run_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute'",
                (ROUND,),
            )
            terminal_judgments_before = cursor.fetchone()[0]
            rerun315_history_before = _round_history(cursor, ROUND + "-r315archive")
            cursor.execute(
                "SELECT rewards_enabled,effective_reward_epoch,reward_basis_hash,"
                "reward_basis_doc,signing_key_doc,reward_activated_at,king_outcome,"
                "king_hotkey,king_start_epoch,promotion_required,promotion_doc,"
                "baseline_promoted_at,confirmation_bank_ref,confirmation_bank_hash,"
                "confirmation_cohort,configuration_doc-ARRAY['schedule','scorer_image_digest',"
                "'scorer_image_reference'] FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            authority_and_policy_before = cursor.fetchone()
            behavior_seals_before = _function_seals(cursor)
            rendered, old_scorer_hash, new_scorer_hash = _render(cursor, schedule)

        attacked = rendered.replace(old_scorer_hash, "0" * 64, 1)
        with pytest.raises(psycopg2.Error, match="Sep19 rerun316 scorer definition differs"):
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
        with pytest.raises(psycopg2.Error, match="Sep19 rerun316 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(attacked)
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute(rendered)
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE status='pending'),"
                "bool_and(assignment_id LIKE '%%:rerun316') "
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
                "SELECT configuration_doc->>'icp_wall_clock_seconds',"
                "configuration_doc->>'execution_icp_cap_microusd',"
                "configuration_doc->>'cost_per_company_microusd',"
                "configuration_doc->'call_quotas'->>'openrouter',"
                "configuration_doc->'schedule' FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                "2700", "4000000", "800000", "200", schedule
            )
            cursor.execute(
                "SELECT s.source_ref,s.source_size_bytes,"
                "s.submission_doc->>'source_sha256',s.submission_doc->>'source_commit',"
                "r.configuration_doc->>'scorer_image_digest',"
                "r.configuration_doc->>'scorer_image_reference' "
                "FROM public.lab_arena_submissions s "
                "JOIN public.lab_arena_rounds r USING(round_id) "
                "WHERE s.round_id=%s AND s.submission_id=%s",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == source_and_scorer_before
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
            )
            definition = cursor.fetchone()[0]
            assert hashlib.sha256(definition.encode()).hexdigest() == new_scorer_hash
            assert "lab_arena_sep19_rerun315_active_valid_v1()" not in definition
            assert ":score:rerun315" not in definition
            assert "lab_arena_sep19_rerun316_active_valid_v1()" in definition
            assert ":score:rerun316" in definition
            cursor.execute(
                "SELECT tgname FROM pg_trigger WHERE tgrelid='public.lab_arena_runs'::regclass "
                "AND NOT tgisinternal AND tgname LIKE 'lab_arena_sep19_rerun%%score_namespace_guard' "
                "ORDER BY tgname"
            )
            assert cursor.fetchall() == [
                ("lab_arena_sep19_rerun316_score_namespace_guard",)
            ]
            assert _function_seals(cursor) == behavior_seals_before
            assert _round_history(cursor, ROUND + "-r315archive") == rerun315_history_before
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE status='accepted'),"
                "count(*) FILTER(WHERE status='failed'),"
                "count(*) FILTER(WHERE terminal_cause='model_error'),"
                "count(*) FILTER(WHERE terminal_cause='provider_error'),"
                "count(DISTINCT assignment_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute' AND submission_id=%s",
                (ROUND + "-r316archive", BASELINE + ":r316archive"),
            )
            assert cursor.fetchone() == (39, 1, 38, 37, 1, 20)
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r)-'round_id'-'submission_id' ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='execute' "
                "AND submission_id=%s",
                (ROUND + "-r316archive", BASELINE + ":r316archive"),
            )
            assert cursor.fetchone()[0] == terminal_baseline_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s AND submission_id=%s",
                (ROUND + "-r316archive", BASELINE + ":r316archive"),
            )
            assert cursor.fetchone()[0] == terminal_baseline_ledger_before
            cursor.execute(
                "SELECT configuration_doc->'archived_execution_judgments' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND + "-r316archive",),
            )
            assert cursor.fetchone()[0] == terminal_judgments_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r316archive"),
            )
            assert cursor.fetchone()[0] == prior_ledger_before
            cursor.execute(rendered)
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep19_rerun316_v1("
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
                "SELECT count(*),count(*) FILTER(WHERE assignment_id LIKE '%%:score:rerun316'),"
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
                "baseline_promoted_at,confirmation_bank_ref,confirmation_bank_hash,"
                "confirmation_cohort,configuration_doc-ARRAY['schedule','scorer_image_digest',"
                "'scorer_image_reference'] FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == authority_and_policy_before
            cursor.execute(
                "SELECT status,jsonb_array_length(publication_doc->'final_ranking') "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == ("published", 5)
            assert _round_history(cursor, ROUND + "-r315archive") == rerun315_history_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r316archive"),
            )
            assert cursor.fetchone()[0] == prior_ledger_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s "
                "AND submission_id<>%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            miner_runs_after = cursor.fetchone()[0]
            assert len(miner_runs_after) == len(miner_runs_before)
            for before, after in zip(miner_runs_before, miner_runs_after):
                for field in ("per_icp_score", "qualification_doc", "updated_at"):
                    before.pop(field, None)
                    after.pop(field, None)
            assert miner_runs_after == miner_runs_before
            published_before = _round_history(cursor, ROUND)
            archive316_before = _round_history(cursor, ROUND + "-r316archive")
        conn.commit()

        with conn.cursor() as cursor:
            cursor.execute(rendered)
            assert _round_history(cursor, ROUND) == published_before
            assert _round_history(cursor, ROUND + "-r316archive") == archive316_before
            assert _round_history(cursor, ROUND + "-r315archive") == rerun315_history_before
    finally:
        conn.close()


def test_terminal315_preimage_requires_exact_shape_active_validity_and_settlement(
    terminal_failure_database,
):
    psycopg2, dsn = terminal_failure_database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_terminal_rerun315(conn)
        with conn.cursor() as cursor:
            rendered, _, _ = _render(cursor, _schedule())
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed' "
                "WHERE run_id=(SELECT run_id FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                "AND status='accepted' ORDER BY run_id LIMIT 1)",
                (ROUND, BASELINE),
            )
            cursor.execute("SET session_replication_role=origin")
        with pytest.raises(psycopg2.Error, match="Sep19 rerun316 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(rendered)
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                """
                WITH run AS (
                  SELECT * FROM public.lab_arena_runs
                  WHERE round_id=%s AND submission_id=%s AND kind='execute'
                  ORDER BY run_id LIMIT 1
                )
                INSERT INTO public.lab_arena_ledger(
                  entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
                  call_identity,provider,operation_id,funding_source,
                  amount_microusd,entry_doc)
                SELECT kinds.kind,miner_hotkey,round_id,submission_id,run_id,stage,
                       'sha256:'||repeat('f',64),'openrouter','openrouter.responses',
                       'host',CASE WHEN kinds.kind='reservation' THEN 4000000 ELSE 0 END,
                       '{}'::jsonb
                FROM run CROSS JOIN unnest(ARRAY['reservation','dispatch']) AS kinds(kind)
                """,
                (ROUND, BASELINE),
            )
            assert cursor.rowcount == 2
            cursor.execute("SET session_replication_role=origin")
            inflight_rendered, _, _ = _render(cursor, _schedule())
        with pytest.raises(psycopg2.Error, match="Sep19 rerun316 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(inflight_rendered)
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET cancel_reason='tampered' "
                "WHERE round_id=%s",
                (ROUND + "-r315archive",),
            )
            cursor.execute("SET session_replication_role=origin")
            cursor.execute("SELECT public.lab_arena_sep19_rerun315_active_valid_v1()")
            assert cursor.fetchone()[0] is False
        with pytest.raises(psycopg2.Error, match="Sep19 rerun316 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(rendered)
        conn.rollback()
    finally:
        conn.close()


def test_template_is_sealed_terminal315_only():
    body = TEMPLATE.read_text()
    assert _test_migrations()[-1] == (
        "314-lab-arena-openrouter-web-search-reservation.sql"
    )
    assert "after the Sep19 rerun315 round is terminal" in body
    assert "assignment_id LIKE '%:rerun315'" in body
    assert "assignment_id NOT LIKE '%:score:rerun315'" in body
    assert "assignment_id LIKE '%:recovery309'" not in body
    assert body.count("CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_v2") == 0
    assert "old_guard TEXT:=$guard315$" in body
    assert "old_assignment TEXT:=$assignment315$" in body
    assert "lab_arena_sep19_rerun315_active_valid_v1()" in body
    assert "lab_arena_sep19_rerun316_active_valid_v1()" in body
    assert "DROP TRIGGER IF EXISTS lab_arena_sep19_rerun315_score_namespace_guard" in body
    assert "active_round.status IS DISTINCT FROM '__TERMINAL_STATUS__'" in body
    assert "active_round.status<>'cancelled'" in body
    assert "active_round.cancel_reason IS DISTINCT FROM '__TERMINAL_CANCEL_REASON__'" in body
    assert "__TERMINAL_BASELINE_ACCEPTED_RUN_COUNT__" in body
    assert "__TERMINAL_BASELINE_FAILED_RUN_COUNT__" in body
    assert "__TERMINAL_BASELINE_LEDGER_COUNT__" in body
    assert "__TERMINAL_SCORE_LEDGER_COUNT__" in body
    assert "terminal_cause='model_error')<>25" not in body
    assert "terminal_cause='provider_error')<>15" not in body
    assert "__SCORING_DEFINITION_SHA256__" in body
    assert "__PATCHED_SCORING_DEFINITION_SHA256__" in body
    assert "__NEW_SOURCE_COMMIT__" in body
    assert "__NEW_SCORER_DIGEST__" in body
    assert "__RERUN_SCHEDULE_JSON__" in body
    assert "__TERMINAL_ACCEPTED_EXECUTE_RUN_COUNT__" in body
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert "lab_arena_accepted_weight_states" in body
    assert "company_quality_policy" in body
