"""PostgreSQL proof for the sealed Sep19 rerun324 migration."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena import (
    sep19_terminal322_completed_429_retry_baseline_rerun323_postgres_test as prior,
)
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


TEMPLATE = Path(__file__).resolve().parents[2] / "scripts" / (
    "324-arena-2026-09-19-terminal323-latest-provider-results-baseline-rerun.sql.template"
)
RENDERED = TEMPLATE.with_suffix("")
# These constants bind the rendered migration to the release owner's captured
# terminal preimage and approved source identity. They remain unset until
# rerun323 is terminal and the release owner performs the exact capture/render.
RENDERED_SHA256: str | None = "837e4dbc5b5f12fbd055578448056188d52d52176d7a7a92f86469a1cad1c8be"
TERMINAL_CAPTURE_SHA256: str | None = "d729b51eca86b22fa426083560ca66c5418eabd72d733fe6529e5aaa7a992a44"
TERMINAL_MARKERS: dict[str, str | None] = {'__PATCHED_SCORING_DEFINITION_SHA256__': '3223be439b7dde72298dcccfe0fc7150aefe50358da1ee7e46a2df5d62dda441',
 '__SCORING_DEFINITION_SHA256__': 'bb3ba46e0d4412250eb43182995920e500616a5950c96fbf4b2809f6c077589e',
 '__TERMINAL_ACCEPTED_EXECUTE_RUN_COUNT__': '92',
 '__TERMINAL_BANK_STATE_SHA256__': '1af19ff80c88e0fa121d57fa11ea6af14b35220017b0d11a099c70f6b60b314f',
 '__TERMINAL_BASELINE_ACCEPTED_RUN_COUNT__': '12',
 '__TERMINAL_BASELINE_EXECUTE_LEDGER_COUNT__': '10278',
 '__TERMINAL_BASELINE_FAILED_RUN_COUNT__': '17',
 '__TERMINAL_BASELINE_LEDGER_COUNT__': '10278',
 '__TERMINAL_BASELINE_RUN_COUNT__': '29',
 '__TERMINAL_BASELINE_SHA256__': 'aa8d38dcaf010b3be1558cdf4a4d344524be7e45247617244c9d60c6e1f9f177',
 '__TERMINAL_CANCEL_REASON_SQL__': "'execution_incomplete:stage1:6'",
 '__TERMINAL_EXECUTE_RUN_COUNT__': '109',
 '__TERMINAL_LEDGER_SHA256__': '077828ce1ab045490a27013c098114758224b2e200438538854869eb43fb7004',
 '__TERMINAL_MINER_SUBMISSIONS_SHA256__': '781027021d3cadc5f2c0540d9f24a67f65cb25126912ef3544867d5a70153297',
 '__TERMINAL_REWARD_AUTHORITY_SHA256__': '22a411e2edcd506ab991f1e65db14f92dd83bfbfeedf7c95b4d478b2d880de2b',
 '__TERMINAL_ROUND_SHA256__': '4328ecf36bff1ca09c10fa0a655aa033347ff69b2748d67bed4fc7b4ed32c045',
 '__TERMINAL_RUNS_SHA256__': 'dffe6616fa5c2733e32788d693d8bbb342b78319018cdef7aef90193e57fae3e',
 '__TERMINAL_SCORE_LEDGER_COUNT__': '0',
 '__TERMINAL_SCORE_RUN_COUNT__': '0',
 '__TERMINAL_STATUS__': 'cancelled'}
EXACT_SOURCE_REF = (
    "arena/arena-2026-09-19/sources/"
    "baseline-2026-09-19-rerun324-6ebb34ef.tar.gz"
)
EXACT_SOURCE_SIZE = 853_258
EXACT_SOURCE_SHA = (
    "5c2a114cc76b4c4ed8e571babc6997daea8945ef97b50d9c8baf0cdf9e401d97"
)
EXACT_SOURCE_COMMIT = "6ebb34efe7a1179de0b6f24cdd2c7e3c6962f470"
# The release owner selects the future window during terminal render.
EXACT_SCHEDULE: dict[str, str] | None = {'submission_open': '2026-09-18T00:00:00Z',
 'submission_cutoff': '2026-09-19T00:00:00Z',
 'benchmark_deadline': '2026-09-19T23:00:00Z',
 'stage_1_start': '2026-09-19T23:00:01Z',
 'stage_1_close': '2026-09-20T02:00:01Z',
 'stage_1_scoring_close': '2026-09-20T05:00:00Z',
 'stage_2_start': '2026-09-20T05:00:01Z',
 'stage_2_close': '2026-09-20T08:00:01Z',
 'final_scoring_close': '2026-09-20T11:00:00Z',
 'publication_deadline': '2026-09-20T11:00:01Z'}
rerun310 = prior.rerun310
ROUND = prior.ROUND
BASELINE = prior.BASELINE
# These identities exist only inside the disposable fixture. Production source
# identity and terminal seals remain mandatory template render inputs.
NEW_SOURCE_REF = (
    "arena/arena-2026-09-19/sources/"
    "baseline-rerun324-synthetic.tar.gz"
)
NEW_SOURCE_SIZE = 812_345
NEW_SOURCE_SHA = "ab" * 32
NEW_SOURCE_COMMIT = "cd" * 20
NEW_SCORER_DIGEST = (
    "sha256:e2fa040d8b1398fad2a802c7dd80a1c709fe68efc485115aae45211b8c489889"
)
NEW_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + NEW_SCORER_DIGEST
)
FAILED_RESPONSE_HOLDS = (
    {
        "icp_position": 5,
        "attempt": 2,
        "call_identity": "sha256:" + "1" * 64,
        "generation_id": "gen-rerun323-web-429-a",
        "credential_fingerprint": "sha256:" + "d" * 64,
        "held_microusd": 3_219_310,
    },
    {
        "icp_position": 1,
        "attempt": 1,
        "call_identity": "sha256:" + "2" * 64,
        "generation_id": "gen-rerun323-web-429-b",
        "credential_fingerprint": "sha256:" + "e" * 64,
        "held_microusd": 2_720_364,
    },
    {
        "icp_position": 2,
        "attempt": 2,
        "call_identity": "sha256:" + "3" * 64,
        "generation_id": "gen-rerun323-web-429-confirmed-cost",
        "credential_fingerprint": "sha256:" + "f" * 64,
        "held_microusd": 0,
    },
)


def _test_migrations() -> tuple[str, ...]:
    return prior._test_migrations()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(_test_migrations())


@pytest.fixture(scope="module")
def terminal_preimage_database():
    for psycopg2, dsn in database_with_lab_arena_migration(_test_migrations()):
        conn = psycopg2.connect(**dsn)
        try:
            _seed_terminal_rerun323(conn)
        finally:
            conn.close()
        yield psycopg2, dsn




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


def _assert_exact_terminal323_render(body: str) -> None:
    assert re.search(r"__[A-Z0-9_]+__", body) is None
    assert all(value is not None for value in TERMINAL_MARKERS.values())
    for value in TERMINAL_MARKERS.values():
        assert value in body
    assert (
        "active_round.status IS DISTINCT FROM "
        f"'{TERMINAL_MARKERS['__TERMINAL_STATUS__']}'"
    ) in body
    assert (
        "active_round.cancel_reason IS DISTINCT FROM "
        f"{TERMINAL_MARKERS['__TERMINAL_CANCEL_REASON_SQL__']}"
    ) in body
    assert (
        "kind='execute')<>"
        f"{TERMINAL_MARKERS['__TERMINAL_EXECUTE_RUN_COUNT__']}"
    ) in body
    assert (
        "kind='score')<>"
        f"{TERMINAL_MARKERS['__TERMINAL_SCORE_RUN_COUNT__']}"
    ) in body
    assert (
        "moved_baseline_runs<>"
        f"{TERMINAL_MARKERS['__TERMINAL_BASELINE_RUN_COUNT__']}"
    ) in body
    assert (
        "moved_baseline_ledger<>"
        f"{TERMINAL_MARKERS['__TERMINAL_BASELINE_EXECUTE_LEDGER_COUNT__']}"
    ) in body


def _sha(cursor, query: str, parameters=()) -> str:
    cursor.execute(query, parameters)
    return cursor.fetchone()[0]


def _template_block(name: str) -> str:
    body = TEMPLATE.read_text()
    matches = re.findall(rf"\${re.escape(name)}\$(.*?)\${re.escape(name)}\$", body, re.DOTALL)
    assert len(matches) == 1
    return matches[0]


def _scorer_patch(definition: str) -> str:
    old_guard = _template_block("guard323")
    new_guard = _template_block("new_guard")
    old_assignment = _template_block("assignment323")
    new_assignment = _template_block("new_assignment")
    assert definition.count(old_guard) == 1
    assert definition.count(old_assignment) == 1
    return definition.replace(old_guard, new_guard).replace(old_assignment, new_assignment)


def _seed_terminal_rerun323(conn, *, terminal_status: str = "cancelled") -> None:
    assert terminal_status in {"published", "cancelled"}
    prior._seed_terminal_rerun322(conn)
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
                  THEN 'arena/output/rerun323-accepted.json' ELSE NULL END
            WHERE round_id=%s AND kind='execute' AND submission_id=%s
              AND assignment_id LIKE '%%:rerun323' AND attempt=1
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
              AND assignment_id LIKE '%%:rerun323' AND attempt=1 AND icp_position>0
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
                     ':score:rerun323:1',
                   execute.round_id||':'||execute.submission_id||':'||
                     execute.stage::text||':'||execute.icp_position::text||
                     ':score:rerun323',
                   execute.round_id,execute.submission_id,execute.miner_hotkey,
                   execute.stage,execute.icp_position,1,'score','accepted',
                   execute.stage_generation,execute.miner_hotkey,'accepted',
                   'arena/score/rerun323-baseline-accepted.json',execute.run_id,
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
                       ('rerun323-paid-'||n::text)::bytea,'sha256'),'hex') AS identity,
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
                   'sha256:'||encode(extensions.digest(
                     'terminal323-score-paid-call'::bytea,'sha256'),'hex'),
                   'openrouter','openrouter.responses','host',
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
                  'fixture','terminal323-published',
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
            SELECT public.lab_arena_sep19_rerun323_active_valid_v1(),
                   count(*) FILTER(WHERE r.status='accepted'),
                   count(*) FILTER(WHERE r.status='failed'),
                   count(*) FILTER(WHERE r.terminal_cause='model_error'),
                   count(*) FILTER(WHERE r.terminal_cause='provider_error'),
                   count(DISTINCT r.assignment_id)
            FROM public.lab_arena_runs r
            WHERE r.round_id=%s AND r.kind='execute' AND r.submission_id=%s
            GROUP BY public.lab_arena_sep19_rerun323_active_valid_v1()
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


def test_terminal323_rerun_executes_scores_and_publishes_with_preservation(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_terminal_rerun323(conn, terminal_status="published")
        _seed_failed_responses_holds(conn)
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
                (ROUND, ROUND + "-r324archive"),
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
            assert len(terminal_baseline_ledger_before) == 18
            hold_identities = [
                call["call_identity"] for call in FAILED_RESPONSE_HOLDS
            ]
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' "
                "ORDER BY entry_id) FROM public.lab_arena_ledger l "
                "WHERE round_id=%s AND submission_id=%s "
                "AND call_identity = ANY(%s)",
                (ROUND, BASELINE, hold_identities),
            )
            terminal_unknown_rows_before = cursor.fetchone()[0]
            assert len(terminal_unknown_rows_before) == 9
            cursor.execute(
                "SELECT jsonb_agg(jsonb_build_object("
                "'run_id',run_id,'per_icp_score',per_icp_score,"
                "'qualification_doc',qualification_doc,'updated_at',updated_at) "
                "ORDER BY run_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute'",
                (ROUND,),
            )
            terminal_judgments_before = cursor.fetchone()[0]
            rerun323_history_before = _round_history(cursor, ROUND + "-r323archive")
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
            assert terminal_publication_before[2]["fixture"] == "terminal323-published"
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
        with pytest.raises(psycopg2.Error, match="Sep19 rerun324 scorer definition differs"):
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
        with pytest.raises(psycopg2.Error, match="Sep19 rerun324 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(attacked)
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute(rendered)
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE status='pending'),"
                "bool_and(assignment_id LIKE '%%:rerun324') "
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
            assert "lab_arena_sep19_rerun323_active_valid_v1()" not in definition
            assert ":score:rerun323" not in definition
            assert "lab_arena_sep19_rerun324_active_valid_v1()" in definition
            assert ":score:rerun324" in definition
            cursor.execute(
                "SELECT tgname FROM pg_trigger WHERE tgrelid='public.lab_arena_runs'::regclass "
                "AND NOT tgisinternal AND tgname LIKE 'lab_arena_sep19_rerun%%score_namespace_guard' "
                "ORDER BY tgname"
            )
            assert cursor.fetchall() == [
                ("lab_arena_sep19_rerun324_score_namespace_guard",)
            ]
            assert _function_seals(cursor) == behavior_seals_before
            assert _round_history(cursor, ROUND + "-r323archive") == rerun323_history_before
            cursor.execute(
                "SELECT publication_doc,published_at,"
                "configuration_doc->>'archived_terminal_status',"
                "configuration_doc->>'archived_terminal_cancel_reason',"
                "configuration_doc->>'archived_terminal_round_sha256' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND + "-r324archive",),
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
                (ROUND + "-r324archive", BASELINE + ":r324archive"),
            )
            assert cursor.fetchone() == source_and_scorer_before[:4]
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE status='accepted'),"
                "count(*) FILTER(WHERE status='failed'),"
                "count(*) FILTER(WHERE terminal_cause='model_error'),"
                "count(*) FILTER(WHERE terminal_cause='provider_error'),"
                "count(DISTINCT assignment_id) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute' AND submission_id=%s",
                (ROUND + "-r324archive", BASELINE + ":r324archive"),
            )
            assert cursor.fetchone() == (39, 1, 38, 37, 1, 20)
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r)-'round_id'-'submission_id' ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='execute' "
                "AND submission_id=%s",
                (ROUND + "-r324archive", BASELINE + ":r324archive"),
            )
            assert cursor.fetchone()[0] == terminal_baseline_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s AND submission_id=%s",
                (ROUND + "-r324archive", BASELINE + ":r324archive"),
            )
            assert cursor.fetchone()[0] == terminal_baseline_ledger_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l)-'round_id'-'submission_id' "
                "ORDER BY entry_id) FROM public.lab_arena_ledger l "
                "WHERE round_id=%s AND submission_id=%s "
                "AND call_identity = ANY(%s)",
                (
                    ROUND + "-r324archive",
                    BASELINE + ":r324archive",
                    hold_identities,
                ),
            )
            assert cursor.fetchone()[0] == terminal_unknown_rows_before
            cursor.execute(
                "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',"
                "'openrouter')",
                (BASELINE + ":r324archive",),
            )
            archived_cost = cursor.fetchone()[0]
            assert archived_cost["inflight_calls"] == 0
            assert archived_cost["success_unresolved_calls"] == 0
            assert archived_cost["uncertain_calls"] == len(FAILED_RESPONSE_HOLDS)
            assert archived_cost["reserved_or_uncertain_microusd"] == sum(
                call["held_microusd"] for call in FAILED_RESPONSE_HOLDS
            )
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s AND kind='score'",
                (ROUND + "-r324archive", BASELINE + ":r324archive"),
            )
            assert cursor.fetchone()[0] == 1
            cursor.execute(
                "SELECT configuration_doc->'archived_execution_judgments' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND + "-r324archive",),
            )
            assert cursor.fetchone()[0] == terminal_judgments_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r324archive"),
            )
            assert cursor.fetchone()[0] == prior_ledger_before
            cursor.execute(rendered)
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep19_rerun324_v1("
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
                "SELECT count(*),count(*) FILTER(WHERE assignment_id LIKE '%%:score:rerun324'),"
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
            assert _round_history(cursor, ROUND + "-r323archive") == rerun323_history_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r324archive"),
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
            archive324_before = _round_history(cursor, ROUND + "-r324archive")
        conn.commit()

        with conn.cursor() as cursor:
            cursor.execute(rendered)
            assert _round_history(cursor, ROUND) == published_before
            assert _round_history(cursor, ROUND + "-r324archive") == archive324_before
            assert _round_history(cursor, ROUND + "-r323archive") == rerun323_history_before
    finally:
        conn.close()


def test_failed_response_fixture_preserves_unknown_cost_without_fee_assumption():
    body = TEMPLATE.read_text()
    assert len(FAILED_RESPONSE_HOLDS) == 3
    assert sorted(call["held_microusd"] for call in FAILED_RESPONSE_HOLDS) == [
        0,
        2_720_364,
        3_219_310,
    ]
    assert len({call["call_identity"] for call in FAILED_RESPONSE_HOLDS}) == 3
    assert body.count("UPDATE public.lab_arena_ledger") == 2
    assert "SET amount_microusd" not in body
    assert "SET entry_doc" not in body
    assert "SET terminal_response" not in body


def test_terminal323_preimage_requires_exact_shape_active_validity_and_settlement(
    terminal_preimage_database,
):
    psycopg2, dsn = terminal_preimage_database
    conn = psycopg2.connect(**dsn)
    try:
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
        with pytest.raises(psycopg2.Error, match="Sep19 rerun324 terminal preimage differs"):
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
        with pytest.raises(psycopg2.Error, match="Sep19 rerun324 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(inflight_rendered)
        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET cancel_reason='tampered' "
                "WHERE round_id=%s",
                (ROUND + "-r323archive",),
            )
            cursor.execute("SET session_replication_role=origin")
            cursor.execute("SELECT public.lab_arena_sep19_rerun323_active_valid_v1()")
            assert cursor.fetchone()[0] is False
        with pytest.raises(psycopg2.Error, match="Sep19 rerun324 terminal preimage differs"):
            with conn.cursor() as cursor:
                cursor.execute(rendered)
        conn.rollback()
    finally:
        conn.close()


def test_template_is_sealed_terminal323_only():
    body = TEMPLATE.read_text()
    assert _test_migrations()[-1] == "321-lab-arena-confirmed-cost-admission.sql"
    assert RENDERED == TEMPLATE.with_suffix("")
    assert RENDERED.exists()
    assert RENDERED.name not in _test_migrations()
    assert TEMPLATE.name not in _test_migrations()
    assert "after the Sep19 rerun323 round is terminal" in body
    assert "assignment_id LIKE '%:rerun323'" in body
    assert "assignment_id NOT LIKE '%:score:rerun323'" in body
    assert "assignment_id LIKE '%:recovery309'" not in body
    assert body.count("CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_v2") == 0
    assert "old_guard TEXT:=$guard323$" in body
    assert "old_assignment TEXT:=$assignment323$" in body
    assert "lab_arena_sep19_rerun323_active_valid_v1()" in body
    assert "lab_arena_sep19_rerun324_active_valid_v1()" in body
    assert "DROP TRIGGER IF EXISTS lab_arena_sep19_rerun323_score_namespace_guard" in body
    assert "AS $score_guard324$" in body and "END $score_guard324$" in body
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
    assert len(set(re.findall(r"__[A-Z0-9_]+__", body))) == 27
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert "lab_arena_accepted_weight_states" in body
    assert "company_quality_policy" in body


def test_exact_render_has_terminal_seals_and_rejects_wrong_preimage_atomically(
    terminal_preimage_database,
):
    assert RENDERED_SHA256 is not None
    assert TERMINAL_CAPTURE_SHA256 is not None
    assert EXACT_SCHEDULE is not None
    raw = RENDERED.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RENDERED_SHA256
    body = raw.decode()
    _assert_exact_terminal323_render(body)
    assert EXACT_SOURCE_COMMIT in body
    assert EXACT_SOURCE_SHA in body
    assert EXACT_SOURCE_REF in body
    assert f"source_size_bytes={EXACT_SOURCE_SIZE}" in body
    assert NEW_SCORER_DIGEST in body
    assert _compact(EXACT_SCHEDULE) in body

    psycopg2, dsn = terminal_preimage_database
    conn = psycopg2.connect(**dsn)
    try:
        with conn.cursor() as cursor:
            before = _round_history(cursor, ROUND)
            previous_archive_before = _round_history(cursor, ROUND + "-r323archive")
            scorer_hash_before = _sha(
                cursor,
                "SELECT encode(extensions.digest(pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure),"
                "'sha256'),'hex')",
            )
        with pytest.raises(
            psycopg2.Error,
            match=(
                "Sep19 rerun324 (scorer definition differs|terminal preimage differs|"
                "admission window closed)"
            ),
        ):
            with conn.cursor() as cursor:
                cursor.execute(body)
        conn.rollback()
        with conn.cursor() as cursor:
            assert _round_history(cursor, ROUND) == before
            assert _round_history(cursor, ROUND + "-r323archive") == (
                previous_archive_before
            )
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND + "-r324archive",),
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                "SELECT to_regprocedure("
                "'public.lab_arena_prepare_sep19_rerun324_v1(bigint,text,text,jsonb,text,text)')"
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
