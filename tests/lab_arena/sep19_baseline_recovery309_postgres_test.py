"""Focused PostgreSQL proof for the sealed Sep19 recovery309 transition."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.lab_arena import sep19_preexecution_source_quota306_postgres_test as migration306
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/309-arena-2026-09-19-baseline-recovery.sql"
ROUND = "arena-2026-09-19"
BASELINE = "baseline-2026-09-19"
SEALED_SCHEDULE = {
    "submission_open": "2026-09-18T00:00:00Z",
    "submission_cutoff": "2026-09-19T00:00:00Z",
    "benchmark_deadline": "2026-09-19T06:00:00Z",
    "stage_1_start": "2026-09-19T06:00:01Z",
    "stage_1_close": "2026-09-19T09:00:01Z",
    "stage_1_scoring_close": "2026-09-19T12:00:00Z",
    "stage_2_start": "2026-09-19T12:00:01Z",
    "stage_2_close": "2026-09-19T15:00:01Z",
    "final_scoring_close": "2026-09-19T18:00:00Z",
    "publication_deadline": "2026-09-19T18:00:01Z",
}


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _compact_schedule(schedule: dict) -> str:
    return json.dumps(schedule, sort_keys=True, separators=(",", ":"))


def _rendered_sql(schedule: dict) -> str:
    """Substitute only the sealed schedule for a reusable future fixture."""
    sql = MIGRATION.read_text()
    sealed = _compact_schedule(SEALED_SCHEDULE)
    assert sql.count(sealed) == 2
    return sql.replace(sealed, _compact_schedule(schedule))


def _function_ddl(schedule: dict) -> str:
    sql = _rendered_sql(schedule)
    return sql.split(
        "SELECT public.lab_arena_prepare_sep19_recovery309_v1(", 1
    )[0] + "\nCOMMIT;\n"


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


def _scoring_definitions(cursor) -> list[tuple[str, str]]:
    cursor.execute(
        """
        SELECT procedure.oid::regprocedure::text,
               pg_get_functiondef(procedure.oid)
        FROM pg_catalog.pg_proc procedure
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid=procedure.pronamespace
        WHERE namespace.nspname='public'
          AND procedure.proname IN (
            'lab_arena_open_scoring',
            'lab_arena_open_scoring_v2',
            'lab_arena_open_scoring_v3',
            'lab_arena_record_run_scores'
          )
        ORDER BY procedure.oid::regprocedure::text
        """
    )
    return cursor.fetchall()


def _integrity_scoring_items(cursor, stage: int) -> list[dict]:
    cursor.execute(
        """
        SELECT jsonb_agg(
          jsonb_build_object(
            'scored_run_id',run.run_id,
            'submission_id',run.submission_id,
            'icp_position',run.icp_position,
            'output_ref',run.output_ref,
            'judgment_cache_key',
              'sha256:'||encode(extensions.digest(run.run_id||':cache','sha256'),'hex'),
            'judgment_input_hash',
              'sha256:'||encode(extensions.digest(run.run_id||':input','sha256'),'hex'),
            'judgment_scope_doc',jsonb_build_object(
              'cache_key',
                'sha256:'||encode(extensions.digest(run.run_id||':cache','sha256'),'hex'),
              'scoring_input_hash',
                'sha256:'||encode(extensions.digest(run.run_id||':input','sha256'),'hex'),
              'round_id',run.round_id,
              'network_name',round.arena_network_name,
              'netuid',round.arena_netuid,
              'integrity_policy','arena_integrity_v1',
              'evaluation_date',round.evaluation_date::text,
              'scorer_image_digest',
                round.configuration_doc->>'scorer_image_digest',
              'scorer_image_reference',
                round.configuration_doc->>'scorer_image_reference'
            ),
            'judgment_group_leader',true,
            'judgment_group_miner_hotkeys',jsonb_build_array(run.miner_hotkey)
          ) ORDER BY run.submission_id,run.icp_position
        )
        FROM public.lab_arena_runs run
        JOIN public.lab_arena_rounds round ON round.round_id=run.round_id
        WHERE run.round_id=%s AND run.stage=%s
          AND run.kind='execute' AND run.status='accepted'
        """,
        (ROUND, stage),
    )
    return cursor.fetchone()[0]


def _scoring_plan(stage: int, items: list[dict]) -> dict:
    return {
        "schema_version": "leadpoet.lab_arena.scoring_plan.v1",
        "round_id": ROUND,
        "stage": stage,
        "work_items": [
            {
                key: item[key]
                for key in (
                    "scored_run_id", "submission_id", "icp_position", "output_ref"
                )
            }
            for item in items
        ],
        "zero_rows": [],
    }


def _seed_terminal(conn):
    migration306.seed(conn)
    migration306.execute(conn)
    migration308 = (
        ROOT / "scripts/308-arena-2026-09-19-preexecution-source-activation.sql"
    ).read_text()
    with conn.cursor() as cursor:
        cursor.execute(migration308)
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            """
            UPDATE public.lab_arena_runs
            SET status='accepted',runner_hotkey=miner_hotkey,
                terminal_cause='accepted',output_ref='arena/test/miner-output.json',
                result_doc='{"schema_version":"test.result.v1"}'::jsonb,
                participation_accepted_at=clock_timestamp()
            WHERE round_id=%s AND submission_id<>%s
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 80
        cursor.execute(
            """
            UPDATE public.lab_arena_runs
            SET status=CASE WHEN icp_position<6 THEN 'accepted' ELSE 'failed' END,
                runner_hotkey=miner_hotkey,
                terminal_cause=CASE WHEN icp_position<6 THEN 'accepted' ELSE 'model_error' END,
                output_ref=CASE WHEN icp_position<6 THEN 'arena/test/baseline-output.json' END,
                result_doc=CASE WHEN icp_position<6
                  THEN '{"schema_version":"test.result.v1"}'::jsonb END,
                participation_accepted_at=CASE WHEN icp_position<6
                  THEN clock_timestamp() END
            WHERE round_id=%s AND submission_id=%s
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 20
        cursor.execute(
            """
            INSERT INTO public.lab_arena_runs(
              run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
              icp_position,attempt,kind,status,stage_generation,runner_hotkey,
              terminal_cause,terminal_doc
            )
            SELECT assignment_id||':2',assignment_id,round_id,submission_id,
                   miner_hotkey,stage,icp_position,2,'execute','failed',
                   stage_generation,miner_hotkey,'model_error','{}'::jsonb
            FROM public.lab_arena_runs
            WHERE round_id=%s AND submission_id=%s AND icp_position BETWEEN 6 AND 15
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 10
        cursor.execute(
            """
            WITH calls AS (
              SELECT 'deepline'::text provider,n,
                     'sha256:'||md5('dl'||n::text)||md5('dl'||n::text) identity,
                     CASE WHEN n=1 THEN 8620000 ELSE 0 END amount
              FROM generate_series(1,371) n
            ), entries AS (
              SELECT provider,n,identity,amount,kind
              FROM calls CROSS JOIN (VALUES('reservation'),('dispatch'),('settlement')) k(kind)
            )
            INSERT INTO public.lab_arena_ledger(
              entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              call_identity,provider,operation_id,funding_source,amount_microusd,
              terminal_response
            )
            SELECT entries.kind,run.miner_hotkey,%s,%s,run.run_id,run.stage,identity,
                   provider,provider||'.execute','host',
                   CASE WHEN entries.kind='settlement' THEN amount ELSE 0 END,
                   CASE WHEN entries.kind='settlement'
                     THEN jsonb_build_object('call_succeeded',n<=366) END
            FROM entries
            CROSS JOIN LATERAL (
              SELECT * FROM public.lab_arena_runs
              WHERE round_id=%s AND submission_id=%s ORDER BY run_id LIMIT 1
            ) run
            """,
            (ROUND, BASELINE, ROUND, BASELINE),
        )
        assert cursor.rowcount == 1113
        cursor.execute(
            """
            WITH calls AS (
              SELECT n,'sha256:'||md5('or'||n::text)||md5('or'||n::text) identity
              FROM generate_series(1,1649) n
            ), entries AS (
              SELECT n,identity,kind FROM calls
              CROSS JOIN (VALUES('reservation'),('dispatch')) k(kind)
              UNION ALL SELECT n,identity,'settlement' FROM calls WHERE n<=1434
              UNION ALL SELECT n,identity,'uncertain' FROM calls WHERE n>1434
            )
            INSERT INTO public.lab_arena_ledger(
              entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              call_identity,provider,operation_id,funding_source,amount_microusd,
              entry_doc,terminal_response
            )
            SELECT entries.kind,run.miner_hotkey,%s,%s,run.run_id,run.stage,identity,
                   'openrouter','openrouter.execute','host',
                   CASE WHEN entries.kind='settlement' AND n=1 THEN 4406527
                        WHEN entries.kind='uncertain' AND n=1435 THEN 23393484 ELSE 0 END,
                   CASE WHEN entries.kind='uncertain'
                     THEN '{"call":{"call_succeeded":false}}'::jsonb ELSE '{}'::jsonb END,
                   CASE WHEN entries.kind='settlement'
                     THEN '{"call_succeeded":true}'::jsonb END
            FROM entries
            CROSS JOIN LATERAL (
              SELECT * FROM public.lab_arena_runs
              WHERE round_id=%s AND submission_id=%s ORDER BY run_id LIMIT 1
            ) run
            """,
            (ROUND, BASELINE, ROUND, BASELINE),
        )
        assert cursor.rowcount == 4947
        cursor.execute(
            """
            INSERT INTO public.lab_arena_ledger(
              entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              call_identity,provider,operation_id,funding_source,amount_microusd
            )
            SELECT 'refusal',run.miner_hotkey,%s,%s,run.run_id,run.stage,
                   'sha256:'||md5('ref'||n::text)||md5('ref'||n::text),
                   'openrouter','openrouter.execute','host',0
            FROM generate_series(1,12) n
            CROSS JOIN LATERAL (
              SELECT * FROM public.lab_arena_runs
              WHERE round_id=%s AND submission_id=%s ORDER BY run_id LIMIT 1
            ) run
            """,
            (ROUND, BASELINE, ROUND, BASELINE),
        )
        cursor.execute(
            """
            WITH calls AS (
              SELECT n,'sha256:'||md5('sd'||n::text)||md5('sd'||n::text) identity
              FROM generate_series(1,10) n
            )
            INSERT INTO public.lab_arena_ledger(
              entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              call_identity,provider,operation_id,funding_source,amount_microusd,
              terminal_response
            )
            SELECT k.kind,run.miner_hotkey,%s,%s,run.run_id,run.stage,identity,
                   'scrapingdog','scrapingdog.execute','host',
                   CASE WHEN k.kind='settlement' AND n=1 THEN 1000 ELSE 0 END,
                   CASE WHEN k.kind='settlement'
                     THEN jsonb_build_object('call_succeeded',n<=4) END
            FROM calls CROSS JOIN (VALUES('reservation'),('dispatch'),('settlement')) k(kind)
            CROSS JOIN LATERAL (
              SELECT * FROM public.lab_arena_runs
              WHERE round_id=%s AND submission_id=%s ORDER BY run_id LIMIT 1
            ) run
            """,
            (ROUND, BASELINE, ROUND, BASELINE),
        )
        cursor.execute(
            """
            INSERT INTO public.lab_arena_ledger(
              entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              call_identity,provider,operation_id,funding_source,amount_microusd
            )
            SELECT 'refusal',run.miner_hotkey,%s,run.submission_id,run.run_id,run.stage,
                   'sha256:'||md5('miner'||n::text)||md5('miner'||n::text),
                   'openrouter','openrouter.execute','host',0
            FROM generate_series(1,4896) n
            CROSS JOIN LATERAL (
              SELECT * FROM public.lab_arena_runs
              WHERE round_id=%s AND submission_id<>%s ORDER BY run_id LIMIT 1
            ) run
            """,
            (ROUND, ROUND, BASELINE),
        )
        cursor.execute(
            """
            UPDATE public.lab_arena_rounds
            SET status='cancelled',cancel_reason='execution_incomplete:stage1:10',
                king_outcome='no_king',
                configuration_doc=configuration_doc||
                  '{"integrity_policy":"arena_integrity_v1",'
                  '"rewards_enabled":true,"schedule":{},'
                  '"scorer_image_digest":'
                  '"sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c",'
                  '"icp_wall_clock_seconds":2700,'
                  '"parallel_twenty_icp_execution":true,'
                  '"stage_1_icp_count":10,"stage_2_icp_count":10,'
                  '"contact_policy":"contacts_v1",'
                  '"scorer_image_reference":"test/scorer@sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c",'
                  '"scorer_policy":{"scoring_adapter_version":'
                  '"qualification_contacts_v3"}}'::jsonb,
                confirmation_bank_ref='arena/test/confirmation-bank.json',
                confirmation_bank_hash='sha256:'||repeat('b',64)
            WHERE round_id=%s
            """,
            (ROUND,),
        )
        cursor.execute("SET session_replication_role=origin")
    conn.commit()


def test_literal_prepare_archives_costs_preserves_miners_and_replays(database):
    psycopg2, dsn = database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_terminal(conn)
        schedule = _schedule()
        with conn.cursor() as cursor:
            scoring_definitions_before = _scoring_definitions(cursor)
            assert len(scoring_definitions_before) == 4
            cursor.execute(
                "SELECT benchmark_ref,configuration_doc-'schedule' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            immutable_round_before = cursor.fetchone()
            cursor.execute(_function_ddl(schedule))
            assert _scoring_definitions(cursor) == scoring_definitions_before
            cursor.execute(
                "SELECT jsonb_build_object('runs',(SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s AND submission_id<>%s),'ledger',(SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l WHERE round_id=%s AND submission_id<>%s))",
                (ROUND, BASELINE, ROUND, BASELINE),
            )
            miners_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',NULL)",
                (BASELINE,),
            )
            fixture_cost = cursor.fetchone()[0]
            assert fixture_cost["uncertain_calls"] == 215, fixture_cost
            assert fixture_cost["settled_microusd"] == 13_027_527, fixture_cost
            assert fixture_cost["reserved_or_uncertain_microusd"] == 23_393_484, fixture_cost
            assert fixture_cost["successful_calls"] == 1_804, fixture_cost
            cursor.execute("BEGIN")
            cursor.execute("SAVEPOINT negative_guard")
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',terminal_cause='model_error' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id=%s AND submission_id<>%s)",
                (ROUND, BASELINE),
            )
            cursor.execute("SET session_replication_role=origin")
            with pytest.raises(psycopg2.Error, match="terminal execution totals differ"):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep19_recovery309_v1(%s::jsonb)",
                        (json.dumps(schedule),),
                )
            cursor.execute("ROLLBACK TO SAVEPOINT negative_guard")
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep19_recovery309_v1(%s::jsonb)",
                (json.dumps(schedule),),
            )
            assert cursor.fetchone()[0]["status"] == "prepared"
            cursor.execute(
                "SELECT pg_sleep(greatest(0,extract(epoch FROM "
                "(%s::timestamptz-clock_timestamp())))+0.05)",
                (schedule["benchmark_deadline"],),
            )
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep19_recovery309_v1(%s::jsonb)",
                (json.dumps(schedule),),
            )
            assert cursor.fetchone()[0]["status"] == "existing"
            assert _scoring_definitions(cursor) == scoring_definitions_before
            cursor.execute(
                "SELECT benchmark_ref,configuration_doc-'schedule' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == immutable_round_before
            cursor.execute(
                "SELECT count(*),count(*) FILTER (WHERE status='pending'),"
                "count(DISTINCT icp_position),min(icp_position),max(icp_position),"
                "bool_and(assignment_id LIKE '%%:recovery309') "
                "FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20, 20, 0, 19, True)
            cursor.execute(
                "SELECT count(*),count(*) FILTER (WHERE status='accepted'),"
                "count(*) FILTER (WHERE status='failed') "
                "FROM public.lab_arena_runs "
                "WHERE round_id='arena-2026-09-19-r309archive' "
                "AND submission_id='baseline-2026-09-19:r309archive' "
                "AND kind='execute'"
            )
            assert cursor.fetchone() == (30, 6, 24)
            cursor.execute(
                "SELECT jsonb_build_object('runs',(SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s AND submission_id<>%s),'ledger',(SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l WHERE round_id=%s AND submission_id<>%s))",
                (ROUND, BASELINE, ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == miners_before
            cursor.execute(
                "SELECT count(*),count(*) FILTER (WHERE entry_kind='uncertain') FROM public.lab_arena_ledger WHERE round_id='arena-2026-09-19-r309archive' AND submission_id='baseline-2026-09-19:r309archive'"
            )
            assert cursor.fetchone() == (6102, 215)
            cursor.execute(
                "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',NULL)",
                (BASELINE,),
            )
            assert cursor.fetchone()[0]["call_count"] == 0
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=miner_hotkey,lease_generation=1,lease_token_hash='sha256:'||repeat('a',64),lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s)",
                (ROUND, BASELINE),
            )
            cursor.execute("SET session_replication_role=origin")
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep19_recovery309_v1(%s::jsonb)",
                (json.dumps(schedule),),
            )
            assert cursor.fetchone()[0]["status"] == "existing"

            # Exercise the unchanged integrity scorer with the original :score
            # assignment IDs after the recovered executions become terminal.
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',runner_hotkey=miner_hotkey,"
                "output_ref='arena/test/recovery309/'||run_id||'.json',"
                "result_doc='{}'::jsonb,participation_accepted_at=clock_timestamp() "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            assert cursor.rowcount == 20
            stage1_items = _integrity_scoring_items(cursor, 1)
            assert len(stage1_items) == 50
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='stage1_closed',"
                "stage1_scoring_plan_doc=%s::jsonb WHERE round_id=%s",
                (json.dumps(_scoring_plan(1, stage1_items)), ROUND),
            )
            cursor.execute("SET session_replication_role=origin")
            cursor.execute(
                "SELECT public.lab_arena_open_scoring_v2(%s,1::smallint,%s::jsonb)",
                (ROUND, json.dumps(stage1_items)),
            )
            assert cursor.fetchone()[0]["assignments"] == 50
            cursor.execute("SET session_replication_role=replica")
            stage2_items = _integrity_scoring_items(cursor, 2)
            assert len(stage2_items) == 50
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='stage2_closed',"
                "stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s",
                (json.dumps(_scoring_plan(2, stage2_items)), ROUND),
            )
            cursor.execute("SET session_replication_role=origin")
            cursor.execute(
                "SELECT public.lab_arena_open_scoring_v2(%s,2::smallint,%s::jsonb)",
                (ROUND, json.dumps(stage2_items)),
            )
            assert cursor.fetchone()[0]["assignments"] == 50
            cursor.execute(
                "SELECT count(*),count(DISTINCT scored_run_id),"
                "bool_and(assignment_id LIKE '%%:score'),"
                "bool_and(assignment_id NOT LIKE '%%:rerun%%') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            assert cursor.fetchone() == (100, 100, True, True)
            cursor.execute(
                "SELECT public.lab_arena_open_scoring_v2(%s,2::smallint,%s::jsonb)",
                (ROUND, json.dumps(stage2_items)),
            )
            assert cursor.fetchone()[0]["status"] == "existing"
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep19_recovery309_v1(%s::jsonb)",
                (json.dumps(schedule),),
            )
            assert cursor.fetchone()[0]["status"] == "existing"
            assert _scoring_definitions(cursor) == scoring_definitions_before
        conn.commit()
    finally:
        conn.close()


def test_public_sql_has_exact_seals_and_no_private_payload():
    body = MIGRATION.read_text()
    assert body.count(_compact_schedule(SEALED_SCHEDULE)) == 2
    assert "__RECOVERY_SCHEDULE_JSON__" not in body
    assert "6102" in body and "23393484" in body and "13027527" in body
    assert "score_namespace', 'score'" in body
    assert "CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring" not in body
    assert "CREATE TRIGGER" not in body
    assert "AND (per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL)" in body
    assert re.search(r"__[A-Z0-9_]+__", body) is None
    for forbidden in (
        '"company_name"', '"company_identity_key"', '"intent_details"',
        "linkedin.com/", "DELETE FROM", "TRUNCATE ",
    ):
        assert forbidden not in body
