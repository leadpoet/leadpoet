"""Disposable PostgreSQL transition proof for October 1 semantic rejudging."""

import json
from pathlib import Path

import pytest

from lab_arena import contact_policy, provider_observations
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena import oct01_corrected_scorer_hold378_postgres_test as held
from tests.lab_arena import oct01_intent_state_hold384_postgres_test as prior


MIGRATION = Path(__file__).parents[2] / 'scripts/385-arena-2026-10-01-intent-state-rejudge.sql'
database = prior.database
ROUND = prior.ROUND
NEW_DIGEST = 'sha256:e2f554c0d9c0f6c3d6ebeef9065da1ef930fc7cca38ebd8406237fbcc46656c9'
NEW_RELEASE = '2564ef035ecfa3a32695d4815b84f6336b01cabd'


def _render(cursor, sql):
    for expected, expression in (
        ('872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19',
         "(SELECT participants FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        ('8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61',
         '(SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260930)'),
        ('1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3',
         "(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='arena-2026-10-01')"),
        ('24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa',
         "(SELECT jsonb_agg(jsonb_build_object('run_id',run_id,'submission_id',submission_id,'stage',stage,'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,'output_ref',output_ref,'stage_generation',stage_generation) ORDER BY icp_position) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='execute')"),
    ):
        assert expected in sql
        sql = sql.replace(expected, held._hash(cursor, expression))
    assert '__' not in sql
    return sql


def _prepare(cursor, *, drain=True):
    hold_sql = prior._prepare(cursor)
    cursor.execute('SET session_replication_role=replica')
    cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='arena/test/score384.json' WHERE round_id=%s AND stage=1 AND kind='score'", (ROUND,))
    cursor.execute("UPDATE public.lab_arena_runs SET per_icp_score=0,qualification_doc='{\"companies\":[]}'::jsonb WHERE round_id=%s AND stage=1 AND kind='execute'", (ROUND,))
    cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1_scored',status_generation=27,stage_generation=22 WHERE round_id=%s", (ROUND,))
    cursor.execute('SET session_replication_role=origin')
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd) "
        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,1,"
        "'sha256:'||encode(extensions.digest('oct01_judge385_fixture','sha256'),'hex'),'deepline','test_judge385','host',100 "
        "FROM public.lab_arena_runs WHERE run_id='score384:0:1'"
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_trajectory_events "
        "(run_id,event_id,round_id,submission_id,miner_hotkey,"
        "runner_hotkey,assignment_id,icp_identifier,stage,icp_position,"
        "attempt,run_kind,model_role,event_kind,occurred_at,content) "
        "SELECT run_id,'00000000-0000-0000-0000-000000000385',round_id,"
        "submission_id,miner_hotkey,miner_hotkey,assignment_id,'oct01-0',"
        "stage,icp_position,attempt,'score','baseline','provider.response',"
        "'2026-10-02T12:00:00Z','{\"operation_id\":\"test_judge385\"}'::jsonb "
        "FROM public.lab_arena_runs WHERE run_id='score384:0:1'"
    )
    cursor.execute(hold_sql)
    return _render(cursor, MIGRATION.read_text())


def test_rejudge_archives_costs_preserves_all_sources_and_resumes(database, monkeypatch):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute("SELECT configuration_doc->'schedule',evaluation_date,icp_set_date "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            schedule_before, evaluation_before, icp_date_before = cursor.fetchone()
            cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                           "WHERE round_id=%s AND stage=2 ORDER BY run_id", (ROUND,))
            miners_before = cursor.fetchall()
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l "
                           "WHERE round_id=%s AND stage=2 ORDER BY entry_id", (ROUND,))
            costs_before = cursor.fetchall()
            cursor.execute("SELECT to_jsonb(e) FROM public.lab_arena_trajectory_events e "
                           "WHERE round_id=%s AND stage=2 ORDER BY trajectory_id", (ROUND,))
            events_before = cursor.fetchall()
            cursor.execute("SELECT run_id,output_ref,result_doc,miner_hotkey "
                           "FROM public.lab_arena_runs WHERE round_id=%s AND stage=1 "
                           "AND kind='execute' ORDER BY icp_position", (ROUND,))
            baseline_source_before = cursor.fetchall()
            cursor.execute("SELECT to_jsonb(l)-'round_id'-'submission_id' "
                           "FROM public.lab_arena_ledger l WHERE run_id='score384:0:1'")
            score_cost_before = cursor.fetchall()
            cursor.execute("SELECT to_jsonb(e)-'round_id'-'submission_id' "
                           "FROM public.lab_arena_trajectory_events e WHERE run_id='score384:0:1'")
            score_event_before = cursor.fetchall()
            cursor.execute(sql)
            cursor.execute("SELECT status,status_generation,stage_generation,"
                           "configuration_doc->>'scorer_image_digest' "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            assert cursor.fetchone() == ('stage1_closed', 28, 23, NEW_DIGEST)
            cursor.execute("SELECT configuration_doc->'schedule',evaluation_date,icp_set_date "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            schedule_after, evaluation_after, icp_date_after = cursor.fetchone()
            assert schedule_after == {**schedule_before,
                'stage_1_scoring_close': '2026-10-02T16:00:01Z',
                'stage_2_start': '2026-10-02T16:00:02Z'}
            assert (evaluation_after, icp_date_after) == (evaluation_before, icp_date_before)
            cursor.execute("SELECT operator_paused FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone()[0] is False
            cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                           "WHERE round_id=%s AND stage=2 ORDER BY run_id", (ROUND,))
            assert cursor.fetchall() == miners_before
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l "
                           "WHERE round_id=%s AND stage=2 ORDER BY entry_id", (ROUND,))
            assert cursor.fetchall() == costs_before
            cursor.execute("SELECT to_jsonb(e) FROM public.lab_arena_trajectory_events e "
                           "WHERE round_id=%s AND stage=2 ORDER BY trajectory_id", (ROUND,))
            assert cursor.fetchall() == events_before
            cursor.execute("SELECT run_id,output_ref,result_doc,miner_hotkey "
                           "FROM public.lab_arena_runs WHERE round_id=%s AND stage=1 "
                           "AND kind='execute' ORDER BY icp_position", (ROUND,))
            assert cursor.fetchall() == baseline_source_before
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT count(*),count(*) FILTER(WHERE status='accepted') "
                           "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                           (ROUND+'-r385archive',))
            assert cursor.fetchone() == (10, 10)
            cursor.execute("SELECT to_jsonb(l)-'round_id'-'submission_id' "
                           "FROM public.lab_arena_ledger l WHERE run_id='score384:0:1'")
            assert cursor.fetchall() == score_cost_before
            cursor.execute("SELECT to_jsonb(e)-'round_id'-'submission_id' "
                           "FROM public.lab_arena_trajectory_events e WHERE run_id='score384:0:1'")
            assert cursor.fetchall() == score_event_before
            cursor.execute("SELECT round_id,submission_id FROM public.lab_arena_ledger "
                           "WHERE run_id='score384:0:1'")
            assert cursor.fetchone() == (ROUND+'-r385archive', 'baseline-2026-10-01-r385archive')
            cursor.execute("SELECT count(*),count(*) FILTER(WHERE per_icp_score IS NULL "
                           "AND qualification_doc IS NULL) FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND stage=1 AND kind='execute'", (ROUND,))
            assert cursor.fetchone() == (10, 10)
            cursor.execute(sql)  # Read-only replay after owned hold release.
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    service = object.__new__(ArenaService)
    service._store = store
    service._round = lambda rid: store.get_round(rid)
    service._load_scoring_plan = lambda row, stage: row['stage1_scoring_plan_doc']
    service._require_code_review = lambda *args: None
    service.evaluation_icps = lambda rid: daily_icps()[:10]
    class Objects:
        @staticmethod
        def get_bounded(ref, max_bytes):
            config = store.get_round(ROUND)['configuration_doc']
            return json.dumps({'schema_version': contact_policy.output_schema(config),
                               'companies': []}).encode()
    service._objects = Objects()
    monkeypatch.setattr(provider_observations, 'resolve_observations', lambda *a, **kw: [])
    opened = service.open_scoring(ROUND, 1)
    assert opened['status'] == 'ok' and opened['assignments'] == 10
    scores = store.list_runs(ROUND, stage=1, kind='score')
    assert len(scores) == 10
    assert all(r['assignment_id'].endswith(':score:rerun385') for r in scores)
    assert all(r['judgment_scope_doc']['scorer_image_digest'] == NEW_DIGEST for r in scores)
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/newscore385.json',"
                           "result_doc='{\"terminal_status\":\"accepted\"}'::jsonb "
                           "WHERE round_id=%s AND stage=1 AND kind='score' AND status='pending'", (ROUND,))
            assert cursor.rowcount == 10
            cursor.execute('SET session_replication_role=origin')
    assert store.close_scoring(ROUND, 1)['status'] == 'closed'
    plan = store.get_round(ROUND)['stage1_scoring_plan_doc']
    service._outputs_by_run = lambda rid, stage: {
        item['scored_run_id']: [] for item in plan['work_items']}
    service._verified_breakdowns = lambda run, **kwargs: []
    assert service.score_stage(ROUND, 1)['status'] == 'ok'
    participants = [p for p in store.get_round(ROUND)['participants'] if not p['is_king']]
    resumed = store.open_stage(ROUND, 2, participants, list(range(10)))
    assert resumed['status'] == 'ok' and resumed['resumed'] is True
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(DISTINCT assignment_id),count(*) "
                           "FROM public.lab_arena_runs WHERE round_id=%s AND stage=2", (ROUND,))
            assert cursor.fetchone() == (130, 130)
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                           "AND stage=2 AND status='accepted'", (ROUND,))
            assert cursor.fetchone()[0] == 14
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                           "AND stage=2 AND status='pending' AND stage_generation="
                           "(SELECT stage_generation FROM public.lab_arena_rounds WHERE round_id=%s)",
                           (ROUND, ROUND))
            assert cursor.fetchone()[0] == 116


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_runs SET status='leased' WHERE round_id='arena-2026-10-01' AND stage=2 AND status='pending' AND icp_position=2",
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='foreign' WHERE singleton",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}','\"wrong\"'::jsonb) WHERE round_id='arena-2026-10-01'",
    "UPDATE public.lab_arena_runs SET assignment_id='wrong' WHERE run_id='score384:0:1'",
    "UPDATE public.lab_arena_ledger SET entry_kind='dispatch' WHERE round_id='arena-2026-10-01' AND stage=2",
])
def test_rejudge_fails_closed_on_live_or_changed_state(database, mutation):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute('BEGIN')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute(mutation)
            cursor.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error):
                cursor.execute(sql)
            cursor.execute('ROLLBACK')
            cursor.execute("SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s",
                           (ROUND+'-r385archive',))
            assert cursor.fetchone()[0] == 0


@pytest.mark.parametrize('function,old,new', [
    ('public.lab_arena_open_stage(text,smallint,jsonb,integer[])',
     'Oct01 rerun379 preserved miner execution resume',
     'Oct01 rerun379 preserved miner execution altered'),
    ('public.lab_arena_open_scoring_v2(text,smallint,jsonb)',
     'Oct01 rerun383 score namespace',
     'Oct01 rerun383 score namespace\n    -- Oct01 rerun383 score namespace'),
    ('public.lab_arena_open_scoring_v2(text,smallint,jsonb)',
     "':score:rerun383'", "':score:wrong'"),
    ('public.lab_arena_open_stage(text,smallint,jsonb,integer[])',
     "COUNT(DISTINCT assignment_id) FROM public.lab_arena_runs\n         WHERE round_id=p_round_id AND stage=2 AND kind='execute') <> 130",
     "COUNT(DISTINCT assignment_id) FROM public.lab_arena_runs\n         WHERE round_id=p_round_id AND stage=2 AND kind='execute') <> 129"),
])
def test_rejudge_rejects_changed_or_duplicate_owned_branch(database, function, old, new):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute('BEGIN')
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (function,))
            definition = cursor.fetchone()[0]
            assert old in definition
            cursor.execute(definition.replace(old, new))
            with pytest.raises(psycopg.Error):
                cursor.execute(sql)
            cursor.execute('ROLLBACK')
            cursor.execute("SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s",
                           (ROUND+'-r385archive',))
            assert cursor.fetchone()[0] == 0


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{recovery_score_runs_sha256}','\"wrong\"'::jsonb) WHERE round_id='arena-2026-10-01-r385archive'",
    "UPDATE public.lab_arena_ledger SET amount_microusd=amount_microusd+1 WHERE run_id='score384:0:1'",
])
def test_rejudge_replay_rejects_archive_or_cost_drift(database, mutation):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute(sql)
            cursor.execute('BEGIN')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute(mutation)
            cursor.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error):
                cursor.execute(sql)
            cursor.execute('ROLLBACK')
            cursor.execute(sql)
