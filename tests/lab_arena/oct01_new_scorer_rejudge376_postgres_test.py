"""Disposable PostgreSQL proof of the October 1 saved-output rejudge."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contact_policy, provider_observations, public_dashboard
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import oct01_held_deadline_recovery372_postgres_test as prior
from tests.lab_arena.icp_fixtures import daily_icps


MIGRATION = Path(__file__).parents[2] / "scripts/376-arena-2026-10-01-new-scorer-rejudge.sql"
database = prior.database


def _prepare(cursor):
    recovery_sql, plan, _ = prior._prepare(cursor)
    cursor.execute(
        "UPDATE public.lab_arena_runs SET qualification_doc='{" 
        "\"companies\":[]}'::jsonb WHERE round_id=%s "
        "AND stage=1 AND kind='execute'", (prior.ROUND,)
    )
    assert cursor.rowcount == 10
    cursor.execute(recovery_sql)
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "UPDATE public.lab_arena_rounds SET status='stage1_judged',"
        "status_generation=11,stage_generation=9 WHERE round_id=%s", (prior.ROUND,)
    )
    cursor.execute(
        "UPDATE public.lab_arena_runs SET status='accepted',"
        "terminal_cause='accepted',output_ref='arena/test/score.json',"
        "result_doc='{\"terminal_status\":\"accepted\"}'::jsonb "
        "WHERE round_id=%s AND stage=1 AND kind='score' "
        "AND status='pending'", (prior.ROUND,)
    )
    assert cursor.rowcount == 9
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd) "
        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,1,"
        "'sha256:'||repeat('c',64),'deepline','test_score','host',100 "
        "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' "
        "AND icp_position=1 AND status='accepted'", (prior.ROUND,)
    )
    assert cursor.rowcount == 1
    cursor.execute("SET session_replication_role=origin")
    cursor.execute("SELECT pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef("
                   "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure),"
                   "'sha256'),'hex')")
    function_hash = cursor.fetchone()[0]
    sql = MIGRATION.read_text().replace(
        "20fb97257eb151be8352c2104aa2f1fa75d4577faab37bdd2ef130e682cc7216",
        function_hash,
    )
    cursor.execute("SELECT configuration_doc->>'scorer_image_digest' FROM "
                   "public.lab_arena_rounds WHERE round_id=%s", (prior.ROUND,))
    sql = sql.replace(
        "sha256:f8ab912f739a1c9e30cc33fb4a5f4ea86b7ef4680af571dc203574f6473f13eb",
        cursor.fetchone()[0],
    )
    return sql, plan


def test_rejudge_archives_history_and_reopens_only_baseline_scoring(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, _ = _prepare(cursor)
            cursor.execute("SELECT to_jsonb(r)-'qualification_doc'-'updated_at' "
                           "FROM public.lab_arena_runs r WHERE round_id=%s "
                           "AND kind='execute' ORDER BY run_id", (prior.ROUND,))
            executions_before = [row[0] for row in cursor.fetchall()]
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE "
                           "round_id=%s AND run_id IN (SELECT run_id FROM "
                           "public.lab_arena_runs WHERE round_id=%s AND kind='score')",
                           (prior.ROUND, prior.ROUND))
            score_ledger_count = cursor.fetchone()[0]
            assert score_ledger_count >= 1
            cursor.execute("SELECT count(*) FROM public.lab_arena_trajectory_events "
                           "WHERE round_id=%s AND run_kind='score'", (prior.ROUND,))
            score_event_count = cursor.fetchone()[0]
            assert score_event_count >= 1
            cursor.execute("SELECT benchmark_ref,evaluation_date,icp_set_date,participants "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (prior.ROUND,))
            frozen_before = cursor.fetchone()
            cursor.execute("SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
                           "FROM public.lab_arena_submissions s WHERE round_id=%s",
                           (prior.ROUND,))
            submissions_before = cursor.fetchone()[0]
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE "
                           "round_id=%s AND kind='score'", (prior.ROUND,))
            assert cursor.fetchone()[0] == 21
            cursor.execute(sql)
            cursor.execute("SELECT status,status_generation,stage_generation,"
                           "configuration_doc->>'scorer_image_digest',"
                           "configuration_doc->'schedule' "
                           "FROM public.lab_arena_rounds WHERE round_id=%s",
                           (prior.ROUND,))
            status, status_gen, stage_gen, digest, schedule = cursor.fetchone()
            assert (status, status_gen, stage_gen) == ('stage1_closed', 12, 10)
            assert digest == 'sha256:342645a42b52363cb907c7b48627ead707e09547a7197b4b0d803e7e6e57a7ba'
            assert schedule['stage_1_scoring_close'] == '2026-10-02T14:00:01Z'
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE "
                           "round_id=%s AND kind='score'", (prior.ROUND,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE "
                           "round_id=%s AND kind='score'", (prior.ROUND+'-r376archive',))
            assert cursor.fetchone()[0] == 21
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE "
                           "round_id=%s AND kind='execute' AND stage=1 "
                           "AND per_icp_score IS NULL AND qualification_doc IS NULL",
                           (prior.ROUND,))
            assert cursor.fetchone()[0] == 10
            cursor.execute("SELECT to_jsonb(r)-'qualification_doc'-'updated_at' "
                           "FROM public.lab_arena_runs r WHERE round_id=%s "
                           "AND kind='execute' ORDER BY run_id", (prior.ROUND,))
            assert [row[0] for row in cursor.fetchall()] == executions_before
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE "
                           "round_id=%s AND run_id IN (SELECT run_id FROM "
                           "public.lab_arena_runs WHERE round_id=%s AND kind='score')",
                           (prior.ROUND+'-r376archive', prior.ROUND+'-r376archive'))
            assert cursor.fetchone()[0] == score_ledger_count
            cursor.execute("SELECT count(*) FROM public.lab_arena_trajectory_events "
                           "WHERE round_id=%s AND run_kind='score'",
                           (prior.ROUND+'-r376archive',))
            assert cursor.fetchone()[0] == score_event_count
            cursor.execute("SELECT configuration_doc->>'recovery_score_runs_sha256',"
                           "configuration_doc->>'recovery_score_ledger_sha256',"
                           "configuration_doc->>'recovery_score_events_sha256' "
                           "FROM public.lab_arena_rounds WHERE round_id=%s",
                           (prior.ROUND+'-r376archive',))
            assert all(len(x) == 64 for x in cursor.fetchone())
            cursor.execute("SELECT benchmark_ref,evaluation_date,icp_set_date,participants "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (prior.ROUND,))
            assert cursor.fetchone() == frozen_before
            cursor.execute("SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
                           "FROM public.lab_arena_submissions s WHERE round_id=%s",
                           (prior.ROUND,))
            assert cursor.fetchone()[0] == submissions_before
            cursor.execute(sql)
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    archive = store.get_round(prior.ROUND+'-r376archive')
    assert archive['status'] == 'cancelled'
    assert archive['configuration_doc']['mode'] == 'live'
    assert archive['rewards_enabled'] is False
    assert archive['king_hotkey'] is None
    assert archive['champion_hotkey'] is None
    assert archive['promotion_required'] is False
    assert public_dashboard._is_administrative_archive(archive)
    service = object.__new__(ArenaService)
    service._store = store
    service._config = SimpleNamespace(mode='live', pinned_round_id=None)
    service._chain_scope = lambda: ('finney', 71)
    service._pinned_round_id = lambda: None
    service._round = lambda rid: store.get_round(rid)
    service._public_round_output_policy = lambda config: {}
    assert service.current_round()['round_id'] != archive['round_id']
    assert archive['round_id'] not in {r['round_id'] for r in service.active_rounds()}
    assert service.open_round()['round_id'] != archive['round_id']
    assert archive['round_id'] not in {
        r['round_id'] for r in public_dashboard._recent_competition_rounds(
            service, network_name='finney', netuid=71, limit=20)
    }


def test_rejudge_rejects_wrong_old_judgment_or_status(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, _ = _prepare(cursor)
            for mutation in (
                "UPDATE public.lab_arena_rounds SET status='stage1_scored' "
                "WHERE round_id='arena-2026-10-01'",
                "UPDATE public.lab_arena_runs SET per_icp_score=1 "
                "WHERE round_id='arena-2026-10-01' AND kind='execute' "
                "AND icp_position=0",
                "UPDATE public.lab_arena_runs SET qualification_doc=NULL "
                "WHERE round_id='arena-2026-10-01' AND kind='execute' "
                "AND icp_position=0",
                "UPDATE public.lab_arena_runs SET status='failed' "
                "WHERE round_id='arena-2026-10-01' AND kind='score' "
                "AND status='accepted' AND icp_position=0",
            ):
                cursor.execute('BEGIN')
                cursor.execute('SET LOCAL session_replication_role=replica')
                cursor.execute(mutation)
                cursor.execute('SET LOCAL session_replication_role=origin')
                with pytest.raises(psycopg.Error, match='precondition differs'):
                    cursor.execute(sql)
                cursor.execute('ROLLBACK')


def test_rejudge_replay_rejects_changed_archive(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, _ = _prepare(cursor)
            cursor.execute(sql)
            cursor.execute('BEGIN')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb "
                           "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND kind='score')",
                           (prior.ROUND+'-r376archive',))
            cursor.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error, match='archive replay differs'):
                cursor.execute(sql)
            cursor.execute('ROLLBACK')


def test_canonical_open_scoring_uses_fresh_namespace_and_image_scope(database, monkeypatch):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, plan = _prepare(cursor)
            cursor.execute("SELECT judgment_cache_key FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND kind='score' AND status='accepted' "
                           "ORDER BY icp_position", (prior.ROUND,))
            old_keys = {row[0] for row in cursor.fetchall()}
            cursor.execute(sql)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                           "SET operator_paused=false,pause_reason='' WHERE singleton")
            cursor.execute("SET session_replication_role=origin")
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    service = object.__new__(ArenaService)
    service._store = store
    service._round = lambda rid: store.get_round(rid)
    service._load_scoring_plan = lambda row, stage: plan
    service._require_code_review = lambda *args: None
    service.evaluation_icps = lambda rid: daily_icps()[:10]
    class Objects:
        @staticmethod
        def get_bounded(ref, max_bytes):
            config = store.get_round(prior.ROUND)['configuration_doc']
            return json.dumps({
                'schema_version': contact_policy.output_schema(config),
                'companies': [],
            }).encode()
    service._objects = Objects()
    monkeypatch.setattr(provider_observations, 'resolve_observations', lambda *a, **kw: [])
    result = service.open_scoring(prior.ROUND, 1)
    assert result['status'] == 'ok' and result['assignments'] == 10
    rows = store.list_runs(prior.ROUND, stage=1, kind='score')
    assert len(rows) == 10 and all(row['status'] == 'pending' for row in rows)
    assert all(row['assignment_id'].endswith(':score:rerun376') for row in rows)
    assert all(row['judgment_cache_key'] not in old_keys for row in rows)
    assert all(row['judgment_scope_doc']['scorer_image_digest'] ==
               'sha256:342645a42b52363cb907c7b48627ead707e09547a7197b4b0d803e7e6e57a7ba'
               for row in rows)
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/new-score.json',"
                           "result_doc='{\"terminal_status\":\"accepted\"}'::jsonb "
                           "WHERE round_id=%s AND kind='score' AND status='pending'",
                           (prior.ROUND,))
            assert cursor.rowcount == 10
            cursor.execute("SET session_replication_role=origin")
    assert store.close_scoring(prior.ROUND, 1)['status'] == 'closed'
    service._outputs_by_run = lambda rid, stage: {
        item['scored_run_id']: [] for item in plan['work_items']
    }
    service._verified_breakdowns = lambda run, **kwargs: []
    assert service.score_stage(prior.ROUND, 1)['status'] == 'ok'
    scored = store.list_runs(prior.ROUND, stage=1, kind='execute')
    assert len(scored) == 10 and all(row['per_icp_score'] == 0 for row in scored)
    assert all(row['qualification_doc'] == {'companies': []} for row in scored)
    participants = [p for p in store.get_round(prior.ROUND)['participants'] if not p['is_king']]
    opened = store.open_stage(prior.ROUND, 2, participants, list(range(10)))
    assert opened['status'] == 'ok' and opened['assignments'] == 130
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_runs "
                "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,stage_generation,"
                "scored_run_id,terminal_cause,result_doc) "
                "SELECT assignment_id||':2',assignment_id,round_id,"
                "submission_id,miner_hotkey,stage,icp_position,2,kind,"
                "'failed',stage_generation,scored_run_id,'judge_error',"
                "'{\"terminal_status\":\"failed\"}'::jsonb "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' "
                "AND stage=1 AND icp_position=0 AND attempt=1",
                (prior.ROUND,),
            )
            assert cursor.rowcount == 1
            cursor.execute("SET session_replication_role=origin")
            cursor.execute(sql)
