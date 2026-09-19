"""Disposable PostgreSQL proof for the sealed Sep19 rerun318 template."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena import (
    sep19_terminal316_fixed_baseline_rerun317_postgres_test as prior,
)
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "scripts/318-arena-2026-09-19-terminal317-fixed-baseline-rerun.sql.template"
RENDERED = ROOT / "scripts/318-arena-2026-09-19-terminal317-fixed-baseline-rerun.sql"
RENDERED_SHA256 = "8bf1847c20863f6085d2fd12c67af5242d27bf4b80c59206a757e52d8e0c1353"
rerun310 = prior.rerun310
ROUND = prior.ROUND
BASELINE = prior.BASELINE
# Synthetic values prove that recovery318 binds a fresh source and scorer.
# Production values remain unfilled placeholders in the SQL template.
NEW_SOURCE_REF = "arena/arena-2026-09-19/sources/baseline-rerun318-test.tar.gz"
NEW_SOURCE_SIZE = 700_318
NEW_SOURCE_SHA = "a" * 64
NEW_SOURCE_COMMIT = "b" * 40
NEW_SCORER_DIGEST = "sha256:" + "c" * 64
NEW_SCORER_REFERENCE = "registry.test/judge@" + NEW_SCORER_DIGEST
FAILED_RESPONSE_HOLDS = (
    {
        "icp_position": 5,
        "attempt": 2,
        "call_identity": "sha256:" + "1" * 64,
        "generation_id": "gen-rerun317-web-429-a",
        "credential_fingerprint": "sha256:" + "d" * 64,
        "held_microusd": 3_219_310,
    },
    {
        "icp_position": 1,
        "attempt": 1,
        "call_identity": "sha256:" + "2" * 64,
        "generation_id": "gen-rerun317-web-429-b",
        "credential_fingerprint": "sha256:" + "e" * 64,
        "held_microusd": 2_720_364,
    },
)


def _test_migrations() -> tuple[str, ...]:
    return prior._test_migrations()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(_test_migrations())


@pytest.fixture(scope="module")
def terminal_failure_database():
    yield from database_with_lab_arena_migration(_test_migrations())


@pytest.fixture(scope="module")
def uncertain_archive_database():
    yield from database_with_lab_arena_migration(_test_migrations())


@pytest.fixture(scope="module")
def exact_render_database():
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
    old_guard = _template_block("guard317")
    new_guard = _template_block("new_guard")
    old_assignment = _template_block("assignment317")
    new_assignment = _template_block("new_assignment")
    assert definition.count(old_guard) == 1
    assert definition.count(old_assignment) == 1
    return definition.replace(old_guard, new_guard).replace(old_assignment, new_assignment)


def _seed_terminal_rerun317(conn, *, terminal_status: str = "cancelled") -> None:
    assert terminal_status in {"published", "cancelled"}
    prior._seed_terminal_rerun316(conn)
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
                  THEN 'arena/output/rerun317-accepted.json' ELSE NULL END
            WHERE round_id=%s AND kind='execute' AND submission_id=%s
              AND assignment_id LIKE '%%:rerun317' AND attempt=1
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
              AND assignment_id LIKE '%%:rerun317' AND attempt=1 AND icp_position>0
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 19
        cursor.execute(
            """
            INSERT INTO public.lab_arena_runs(
              run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
              icp_position,attempt,kind,status,stage_generation,runner_hotkey,
              terminal_cause,output_ref,scored_run_id,judgment_cache_key,
              judgment_input_hash,judgment_scope_doc)
            SELECT execute.round_id||':'||execute.submission_id||':'||
                     execute.stage::text||':'||execute.icp_position::text||
                     ':score:rerun317:1',
                   execute.round_id||':'||execute.submission_id||':'||
                     execute.stage::text||':'||execute.icp_position::text||
                     ':score:rerun317',
                   execute.round_id,execute.submission_id,execute.miner_hotkey,
                   execute.stage,execute.icp_position,1,'score','accepted',
                   execute.stage_generation,execute.miner_hotkey,'accepted',
                   'arena/score/rerun317-baseline-accepted.json',execute.run_id,
                   'sha256:'||repeat('6',64),'sha256:'||repeat('7',64),
                   jsonb_build_object(
                     'scorer_image_digest',round.configuration_doc->>'scorer_image_digest',
                     'scorer_image_reference',round.configuration_doc->>'scorer_image_reference')
            FROM public.lab_arena_runs execute
            JOIN public.lab_arena_rounds round USING(round_id)
            WHERE execute.round_id=%s AND execute.submission_id=%s
              AND execute.kind='execute' AND execute.status='accepted'
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 1
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
                       ('rerun317-paid-'||n::text)::bytea,'sha256'),'hex') AS identity,
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
            WITH score AS (
              SELECT * FROM public.lab_arena_runs
              WHERE round_id=%s AND submission_id=%s AND kind='score'
            )
            INSERT INTO public.lab_arena_ledger(
              entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              call_identity,provider,operation_id,funding_source,amount_microusd,
              entry_doc,terminal_response)
            SELECT entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
                   'sha256:'||'56a1e378bec251bb8cb80fb6b5ac7da4aa77988c24614d2524e9a248cb62a381','openrouter','openrouter.responses','host',
                   CASE WHEN entry_kind='reservation' THEN 4000000
                        WHEN entry_kind='settlement' THEN 7777 ELSE 0 END,
                   jsonb_build_object('sealed_fixture',TRUE,'score_call',TRUE,
                                      'sequence',ordinality),
                   CASE WHEN entry_kind='settlement'
                     THEN jsonb_build_object('call_succeeded',TRUE) ELSE NULL END
            FROM score CROSS JOIN unnest(ARRAY['reservation','dispatch','settlement'])
              WITH ORDINALITY AS kinds(entry_kind,ordinality)
            ORDER BY ordinality
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 3
        cursor.execute(
            """
            UPDATE public.lab_arena_rounds
            SET status=%s,
                published_at=CASE WHEN %s='published' THEN clock_timestamp() ELSE NULL END,
                publication_doc=CASE WHEN %s='published' THEN jsonb_build_object(
                  'schema_version','leadpoet.lab_arena.publication.v1',
                  'fixture','terminal317-published',
                  'final_ranking',jsonb_build_array('miner-a','miner-b','miner-c',
                                                    'miner-d','miner-e'))
                  ELSE NULL END,
                cancel_reason=CASE WHEN %s='cancelled'
                  THEN 'execution_incomplete:stage1:5' ELSE NULL END,
                updated_at=clock_timestamp()
            WHERE round_id=%s
            """,
            (terminal_status, terminal_status, terminal_status, terminal_status, ROUND),
        )
        assert cursor.rowcount == 1
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            """
            SELECT public.lab_arena_sep19_rerun317_active_valid_v1(),
                   count(*) FILTER(WHERE r.status='accepted'),
                   count(*) FILTER(WHERE r.status='failed'),
                   count(*) FILTER(WHERE r.terminal_cause='model_error'),
                   count(*) FILTER(WHERE r.terminal_cause='provider_error'),
                   count(DISTINCT r.assignment_id)
            FROM public.lab_arena_runs r
            WHERE r.round_id=%s AND r.kind='execute' AND r.submission_id=%s
            GROUP BY public.lab_arena_sep19_rerun317_active_valid_v1()
            """,
            (ROUND, BASELINE),
        )
        assert cursor.fetchone() == (True, 1, 38, 37, 1, 20)
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
            (ROUND,),
        )
        assert cursor.fetchone()[0] == 1
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s "
            "AND submission_id=%s",
            (ROUND, BASELINE),
        )
        assert cursor.fetchone()[0] == 9
    conn.commit()


def _seed_failed_responses_holds(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        for call in FAILED_RESPONSE_HOLDS:
            cursor.execute(
                """
                WITH run AS (
                  SELECT run_id,miner_hotkey,stage
                  FROM public.lab_arena_runs
                  WHERE round_id=%s AND submission_id=%s AND kind='execute'
                    AND icp_position=%s AND attempt=%s AND status='failed'
                ), entries(entry_kind,ordinality) AS (
                  VALUES ('reservation',1),('dispatch',2),('uncertain',3)
                )
                INSERT INTO public.lab_arena_ledger(
                  entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
                  call_identity,provider,operation_id,funding_source,
                  amount_microusd,entry_doc,terminal_response)
                SELECT entries.entry_kind,run.miner_hotkey,%s,%s,run.run_id,run.stage,
                       %s,'openrouter','openrouter.responses','host',
                       CASE WHEN entries.entry_kind IN('reservation','uncertain')
                         THEN %s ELSE 0 END,
                       CASE WHEN entries.entry_kind='uncertain' THEN
                         jsonb_build_object(
                           'reason','worker_reported',
                           'call',jsonb_build_object(
                             'reason','missing_provider_cost',
                             'call_succeeded',FALSE,
                             'openrouter_generation_id',%s,
                             'credential_fingerprint',%s))
                       ELSE jsonb_build_object(
                         'fixture','failed_responses_429',
                         'sequence',entries.ordinality)
                       END,
                       CASE WHEN entries.entry_kind='uncertain' THEN
                         jsonb_build_object(
                           'status',429,
                           'call_succeeded',FALSE,
                           'body_b64','eyJlcnJvciI6InJhdGVfbGltaXRlZCJ9')
                       ELSE NULL END
                FROM run CROSS JOIN entries
                ORDER BY entries.ordinality
                """,
                (
                    ROUND,
                    BASELINE,
                    call["icp_position"],
                    call["attempt"],
                    ROUND,
                    BASELINE,
                    call["call_identity"],
                    call["held_microusd"],
                    call["generation_id"],
                    call["credential_fingerprint"],
                ),
            )
            assert cursor.rowcount == 3
        cursor.execute("SET session_replication_role=origin")
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
    if terminal_cancel_reason is not None:
        assert re.fullmatch(r"[a-z0-9_:.-]+", terminal_cancel_reason)
    terminal_cancel_reason_sql = (
        "NULL" if terminal_cancel_reason is None else f"'{terminal_cancel_reason}'"
    )
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
        "__TERMINAL_CANCEL_REASON_SQL__": terminal_cancel_reason_sql,
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
        "__TERMINAL_BASELINE_EXECUTE_LEDGER_COUNT__": str(_sha(
            cursor,
            "SELECT count(*) FROM public.lab_arena_ledger ledger "
            "WHERE ledger.round_id=%s AND ledger.submission_id=%s "
            "AND EXISTS(SELECT 1 FROM public.lab_arena_runs run "
            "WHERE run.round_id=%s AND run.kind='execute' "
            "AND run.run_id=ledger.run_id)",
            (ROUND, BASELINE, ROUND),
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


def test_terminal317_rerun_executes_scores_and_publishes_with_preservation(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_terminal_rerun317(conn, terminal_status="published")
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
                prior.NEW_SOURCE_REF,
                prior.NEW_SOURCE_SIZE,
                prior.NEW_SOURCE_SHA,
                prior.NEW_SOURCE_COMMIT,
                prior.NEW_SCORER_DIGEST,
                prior.NEW_SCORER_REFERENCE,
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
                (ROUND, ROUND + "-r318archive"),
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
            assert len(terminal_baseline_ledger_before) == 9
            cursor.execute(
                "SELECT jsonb_agg(jsonb_build_object("
                "'run_id',run_id,'per_icp_score',per_icp_score,"
                "'qualification_doc',qualification_doc,'updated_at',updated_at) "
                "ORDER BY run_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute'",
                (ROUND,),
            )
            terminal_judgments_before = cursor.fetchone()[0]
            rerun317_history_before = _round_history(cursor, ROUND + "-r317archive")
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
            cursor.execute(
                "SELECT status,published_at,publication_doc,cancel_reason "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            terminal_publication_before = cursor.fetchone()
            assert terminal_publication_before[0] == "published"
            assert terminal_publication_before[1] is not None
            assert terminal_publication_before[2]["fixture"] == "terminal317-published"
            assert terminal_publication_before[3] is None
            terminal_round_sha_before = _sha(
                cursor,
                "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') "
                "FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND,),
            )
            behavior_seals_before = _function_seals(cursor)
            rendered, old_scorer_hash, new_scorer_hash = _render(cursor, schedule)

        attacked = rendered.replace(old_scorer_hash, "0" * 64, 1)
        with pytest.raises(psycopg2.Error, match="Sep19 rerun318 scorer definition differs"):
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
        with pytest.raises(psycopg2.Error, match="Sep19 rerun318 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(attacked)
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute(rendered)
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE status='pending'),"
                "bool_and(assignment_id LIKE '%%:rerun318') "
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
            assert cursor.fetchone() == (
                NEW_SOURCE_REF,
                NEW_SOURCE_SIZE,
                NEW_SOURCE_SHA,
                NEW_SOURCE_COMMIT,
                NEW_SCORER_DIGEST,
                NEW_SCORER_REFERENCE,
            )
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
            )
            definition = cursor.fetchone()[0]
            assert hashlib.sha256(definition.encode()).hexdigest() == new_scorer_hash
            assert "lab_arena_sep19_rerun317_active_valid_v1()" not in definition
            assert ":score:rerun317" not in definition
            assert "lab_arena_sep19_rerun318_active_valid_v1()" in definition
            assert ":score:rerun318" in definition
            cursor.execute(
                "SELECT tgname FROM pg_trigger WHERE tgrelid='public.lab_arena_runs'::regclass "
                "AND NOT tgisinternal AND tgname LIKE 'lab_arena_sep19_rerun%%score_namespace_guard' "
                "ORDER BY tgname"
            )
            assert cursor.fetchall() == [
                ("lab_arena_sep19_rerun318_score_namespace_guard",)
            ]
            assert _function_seals(cursor) == behavior_seals_before
            assert _round_history(cursor, ROUND + "-r317archive") == rerun317_history_before
            cursor.execute(
                "SELECT publication_doc,published_at,"
                "configuration_doc->>'archived_terminal_status',"
                "configuration_doc->>'archived_terminal_cancel_reason',"
                "configuration_doc->>'archived_terminal_round_sha256' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND + "-r318archive",),
            )
            archived_publication = cursor.fetchone()
            assert archived_publication == (
                terminal_publication_before[2],
                terminal_publication_before[1],
                "published",
                None,
                terminal_round_sha_before,
            )
            cursor.execute(
                "SELECT source_ref,source_size_bytes,"
                "submission_doc->>'source_sha256',submission_doc->>'source_commit' "
                "FROM public.lab_arena_submissions WHERE round_id=%s AND submission_id=%s",
                (ROUND + "-r318archive", BASELINE + ":r318archive"),
            )
            assert cursor.fetchone() == source_and_scorer_before[:4]
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE status='accepted'),"
                "count(*) FILTER(WHERE status='failed'),"
                "count(*) FILTER(WHERE terminal_cause='model_error'),"
                "count(*) FILTER(WHERE terminal_cause='provider_error'),"
                "count(DISTINCT assignment_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute' AND submission_id=%s",
                (ROUND + "-r318archive", BASELINE + ":r318archive"),
            )
            assert cursor.fetchone() == (39, 1, 38, 37, 1, 20)
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r)-'round_id'-'submission_id' ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='execute' "
                "AND submission_id=%s",
                (ROUND + "-r318archive", BASELINE + ":r318archive"),
            )
            assert cursor.fetchone()[0] == terminal_baseline_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s AND submission_id=%s",
                (ROUND + "-r318archive", BASELINE + ":r318archive"),
            )
            assert cursor.fetchone()[0] == terminal_baseline_ledger_before
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s AND kind='score'",
                (ROUND + "-r318archive", BASELINE + ":r318archive"),
            )
            assert cursor.fetchone()[0] == 1
            cursor.execute(
                "SELECT configuration_doc->'archived_execution_judgments' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND + "-r318archive",),
            )
            assert cursor.fetchone()[0] == terminal_judgments_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r318archive"),
            )
            assert cursor.fetchone()[0] == prior_ledger_before
            cursor.execute(rendered)
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep19_rerun318_v1("
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
                "SELECT count(*),count(*) FILTER(WHERE assignment_id LIKE '%%:score:rerun318'),"
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
            assert _round_history(cursor, ROUND + "-r317archive") == rerun317_history_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r318archive"),
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
            archive318_before = _round_history(cursor, ROUND + "-r318archive")
        conn.commit()

        with conn.cursor() as cursor:
            cursor.execute(rendered)
            assert _round_history(cursor, ROUND) == published_before
            assert _round_history(cursor, ROUND + "-r318archive") == archive318_before
            assert _round_history(cursor, ROUND + "-r317archive") == rerun317_history_before
    finally:
        conn.close()


def test_failed_responses_unknown_costs_archive_with_full_holds_and_no_fee_assumption(
    uncertain_archive_database,
):
    psycopg2, dsn = uncertain_archive_database
    conn = psycopg2.connect(**dsn)
    identities = [call["call_identity"] for call in FAILED_RESPONSE_HOLDS]
    expected_hold = sum(call["held_microusd"] for call in FAILED_RESPONSE_HOLDS)

    def ledger_rows(cursor, round_id, submission_id):
        cursor.execute(
            "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
            "FROM public.lab_arena_ledger l WHERE round_id=%s AND submission_id=%s "
            "AND call_identity IN(%s,%s)",
            (round_id, submission_id, *identities),
        )
        return cursor.fetchone()[0]

    def assert_reconciliation_is_closed(cursor, round_id, calls):
        cursor.execute(
            "SELECT public.lab_arena_list_openrouter_cost_reconciliations_v1("
            "%s,NULL,0,20)",
            (round_id,),
        )
        assert cursor.fetchone()[0] == {"status": "ok", "items": []}
        for call in calls:
            cursor.execute(
                "SELECT public.lab_arena_reconcile_openrouter_cost_v1("
                "%s,%s,%s,%s,%s,%s,0,'0')",
                (
                    round_id,
                    call["run_id"],
                    call["call_identity"],
                    call["uncertain_entry_id"],
                    call["generation_id"],
                    call["credential_fingerprint"],
                ),
            )
            assert cursor.fetchone()[0] == {"status": "stale"}

    try:
        _seed_terminal_rerun317(conn)
        _seed_failed_responses_holds(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',"
                "'openrouter')",
                (BASELINE,),
            )
            before_cost = cursor.fetchone()[0]
            assert before_cost["inflight_calls"] == 0
            assert before_cost["success_unresolved_calls"] == 0
            assert before_cost["uncertain_calls"] == 2
            assert before_cost["reserved_or_uncertain_microusd"] == expected_hold
            cursor.execute(
                "SELECT call_identity,run_id,entry_id,"
                "entry_doc#>>'{call,openrouter_generation_id}',"
                "entry_doc#>>'{call,credential_fingerprint}' "
                "FROM public.lab_arena_ledger WHERE round_id=%s "
                "AND submission_id=%s AND entry_kind='uncertain' "
                "AND call_identity IN(%s,%s) ORDER BY call_identity",
                (ROUND, BASELINE, *identities),
            )
            uncertain_calls = [
                {
                    "call_identity": row[0],
                    "run_id": row[1],
                    "uncertain_entry_id": row[2],
                    "generation_id": row[3],
                    "credential_fingerprint": row[4],
                }
                for row in cursor.fetchall()
            ]
            assert len(uncertain_calls) == 2
            before_rows = ledger_rows(cursor, ROUND, BASELINE)
            assert len(before_rows) == 6
            assert [row["entry_kind"] for row in before_rows].count("settlement") == 0
            assert_reconciliation_is_closed(cursor, ROUND, uncertain_calls)
            rendered, _, _ = _render(cursor, _schedule())
            cursor.execute(rendered)

            archive_round = ROUND + "-r318archive"
            archive_submission = BASELINE + ":r318archive"
            cursor.execute(
                "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',"
                "'openrouter')",
                (archive_submission,),
            )
            archive_cost = cursor.fetchone()[0]
            assert archive_cost["inflight_calls"] == 0
            assert archive_cost["success_unresolved_calls"] == 0
            assert archive_cost["uncertain_calls"] == 2
            assert archive_cost["reserved_or_uncertain_microusd"] == expected_hold
            after_rows = ledger_rows(cursor, archive_round, archive_submission)
            assert len(after_rows) == 6
            for before, after in zip(before_rows, after_rows):
                assert after.pop("round_id") == archive_round
                assert after.pop("submission_id") == archive_submission
                assert before.pop("round_id") == ROUND
                assert before.pop("submission_id") == BASELINE
                assert after == before
            cursor.execute(
                "SELECT call_identity,entry_kind,amount_microusd,"
                "terminal_response ? 'provider_cost',terminal_response ? 'billing' "
                "FROM public.lab_arena_ledger WHERE round_id=%s "
                "AND submission_id=%s AND call_identity IN(%s,%s) "
                "AND entry_id IN(SELECT max(entry_id) FROM public.lab_arena_ledger "
                "WHERE call_identity IN(%s,%s) GROUP BY call_identity) "
                "ORDER BY call_identity",
                (
                    archive_round,
                    archive_submission,
                    *identities,
                    *identities,
                ),
            )
            assert cursor.fetchall() == [
                (identities[0], "uncertain", 3_219_310, False, False),
                (identities[1], "uncertain", 2_720_364, False, False),
            ]
            cursor.execute(
                "SELECT public.lab_arena_sep19_rerun318_archive_valid_v1(),"
                "public.lab_arena_sep19_rerun318_active_valid_v1()"
            )
            assert cursor.fetchone() == (True, True)
            assert_reconciliation_is_closed(cursor, archive_round, uncertain_calls)
            assert_reconciliation_is_closed(cursor, ROUND, uncertain_calls)

            archive_before_reapply = ledger_rows(
                cursor, archive_round, archive_submission
            )
            cursor.execute(rendered)
            assert ledger_rows(cursor, archive_round, archive_submission) == (
                archive_before_reapply
            )
        conn.commit()

        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd,"
                "entry_doc,terminal_response) SELECT 'settlement',miner_hotkey,"
                "round_id,submission_id,run_id,stage,call_identity,provider,operation_id,"
                "funding_source,0,jsonb_build_object('tampered_append',true),"
                "jsonb_build_object('call_succeeded',false) "
                "FROM public.lab_arena_ledger WHERE round_id=%s AND call_identity=%s "
                "AND entry_kind='uncertain'",
                (ROUND + "-r318archive", identities[0]),
            )
            assert cursor.rowcount == 1
            cursor.execute("SET session_replication_role=origin")
            cursor.execute(
                "SELECT public.lab_arena_sep19_rerun318_archive_valid_v1(),"
                "public.lab_arena_sep19_rerun318_active_valid_v1()"
            )
            assert cursor.fetchone() == (False, False)
        with pytest.raises(psycopg2.Error, match="existing Sep19 rerun318 differs"):
            with conn.cursor() as cursor:
                cursor.execute(rendered)
        conn.rollback()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_sep19_rerun318_archive_valid_v1(),"
                "public.lab_arena_sep19_rerun318_active_valid_v1()"
            )
            assert cursor.fetchone() == (True, True)
    finally:
        conn.close()


def test_terminal317_preimage_requires_exact_shape_active_validity_and_settlement(
    terminal_failure_database,
):
    psycopg2, dsn = terminal_failure_database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_terminal_rerun317(conn)
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
        with pytest.raises(psycopg2.Error, match="Sep19 rerun318 terminal preimage differs"):
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
        with pytest.raises(psycopg2.Error, match="Sep19 rerun318 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(inflight_rendered)
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET cancel_reason='tampered' "
                "WHERE round_id=%s",
                (ROUND + "-r317archive",),
            )
            cursor.execute("SET session_replication_role=origin")
            cursor.execute("SELECT public.lab_arena_sep19_rerun317_active_valid_v1()")
            assert cursor.fetchone()[0] is False
        with pytest.raises(psycopg2.Error, match="Sep19 rerun318 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(rendered)
        conn.rollback()

        with conn.cursor() as cursor:
            cancelled_rendered, _, _ = _render(cursor, _schedule())
            cursor.execute(cancelled_rendered)
            cursor.execute(
                "SELECT publication_doc,published_at,"
                "configuration_doc->>'archived_terminal_status',"
                "configuration_doc->>'archived_terminal_cancel_reason' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND + "-r318archive",),
            )
            assert cursor.fetchone() == (
                None,
                None,
                "cancelled",
                "execution_incomplete:stage1:5",
            )
    finally:
        conn.close()


def test_template_is_sealed_terminal317_only():
    body = TEMPLATE.read_text()
    assert _test_migrations()[-1] == (
        "314-lab-arena-openrouter-web-search-reservation.sql"
    )
    assert RENDERED == TEMPLATE.with_suffix("")
    assert RENDERED.exists()
    assert TEMPLATE.name not in _test_migrations()
    assert "after the Sep19 rerun317 round is terminal" in body
    assert "assignment_id LIKE '%:rerun317'" in body
    assert "assignment_id NOT LIKE '%:score:rerun317'" in body
    assert "assignment_id LIKE '%:recovery309'" not in body
    assert body.count("CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_v2") == 0
    assert "old_guard TEXT:=$guard317$" in body
    assert "old_assignment TEXT:=$assignment317$" in body
    assert "lab_arena_sep19_rerun317_active_valid_v1()" in body
    assert "lab_arena_sep19_rerun318_active_valid_v1()" in body
    assert "DROP TRIGGER IF EXISTS lab_arena_sep19_rerun317_score_namespace_guard" in body
    assert "AS $score_guard318$" in body and "END $score_guard318$" in body
    assert "active_round.status IS DISTINCT FROM '__TERMINAL_STATUS__'" in body
    assert "active_round.status NOT IN('published','cancelled')" in body
    assert "active_round.cancel_reason IS DISTINCT FROM __TERMINAL_CANCEL_REASON_SQL__" in body
    assert "__TERMINAL_BASELINE_ACCEPTED_RUN_COUNT__" in body
    assert "__TERMINAL_BASELINE_FAILED_RUN_COUNT__" in body
    assert "__TERMINAL_BASELINE_LEDGER_COUNT__" in body
    assert "__TERMINAL_BASELINE_EXECUTE_LEDGER_COUNT__" in body
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


def test_exact_render_has_terminal_seals_and_rejects_wrong_preimage_atomically(
    exact_render_database,
):
    raw = RENDERED.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RENDERED_SHA256
    body = raw.decode()
    assert re.search(r"__[A-Z0-9_]+__", body) is None
    assert "6d6cd05979153e09d7d00d3b2bf04990fb023d3b" in body
    assert "5da106d1f3b5078460aa43297ead185f1a1257b75c6c30a1effd3b6464cdf5ff" in body
    assert "baseline-2026-09-19-rerun318-6d6cd059.tar.gz" in body
    assert "source_size_bytes=793558" in body
    assert "sha256:e2fa040d8b1398fad2a802c7dd80a1c709fe68efc485115aae45211b8c489889" in body
    assert '"benchmark_deadline":"2026-09-19T14:00:00Z"' in body
    assert "active_round.status IS DISTINCT FROM 'cancelled'" in body
    assert "active_round.cancel_reason IS DISTINCT FROM 'operator'" in body
    assert "kind='execute')<>110" in body
    assert "kind='score')<>0" in body
    assert "moved_baseline_runs<>30" in body
    assert "moved_baseline_ledger<>6579" in body
    for terminal_seal in (
        "edd40e5004b4b654febc4733f94e62b5e82d9b7f69720035f3ce20a7c6fcad2e",
        "adee534ed61bb1ea0761142d66d22d2edef6de2b89cd73d1b03a5f5a1e675f9f",
        "781027021d3cadc5f2c0540d9f24a67f65cb25126912ef3544867d5a70153297",
        "a40087077d409c804d044bdbfaa04b93bbc72f6dc93ccf61c76dca85ffc478f1",
        "3bc4b1624c68abc3c10cfbea4ddb296969e3c04d6ade07a6b96976f0660c4129",
        "1af19ff80c88e0fa121d57fa11ea6af14b35220017b0d11a099c70f6b60b314f",
        "22a411e2edcd506ab991f1e65db14f92dd83bfbfeedf7c95b4d478b2d880de2b",
        "1bf82d9e27e385424ad96d0552573d63ecc84e55ea04ec81bcd8d74e07c07991",
        "937c57a010c01a3336165f052a65295fe4fef0fab887ce3babba1c82480d95d7",
    ):
        assert terminal_seal in body

    psycopg2, dsn = exact_render_database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_terminal_rerun317(conn)
        with conn.cursor() as cursor:
            before = _round_history(cursor, ROUND)
            previous_archive_before = _round_history(cursor, ROUND + "-r317archive")
            scorer_hash_before = _sha(
                cursor,
                "SELECT encode(extensions.digest(pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure),"
                "'sha256'),'hex')",
            )
        with pytest.raises(
            psycopg2.Error,
            match=(
                "Sep19 rerun318 (scorer definition differs|terminal preimage differs|"
                "admission window closed)"
            ),
        ):
            with conn.cursor() as cursor:
                cursor.execute(body)
        conn.rollback()
        with conn.cursor() as cursor:
            assert _round_history(cursor, ROUND) == before
            assert _round_history(cursor, ROUND + "-r317archive") == (
                previous_archive_before
            )
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND + "-r318archive",),
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                "SELECT to_regprocedure("
                "'public.lab_arena_prepare_sep19_rerun318_v1(bigint,text,text,jsonb,text,text)')"
            )
            assert cursor.fetchone()[0] is None
            assert _sha(
                cursor,
                "SELECT encode(extensions.digest(pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure),"
                "'sha256'),'hex')",
            ) == scorer_hash_before
    finally:
        conn.close()
