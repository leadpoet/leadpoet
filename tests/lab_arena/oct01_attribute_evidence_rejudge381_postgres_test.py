"""Disposable PostgreSQL proof of the October 1 attribute-evidence rejudge draft."""

import json
from pathlib import Path

import pytest

from lab_arena import contact_policy, provider_observations
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena import oct01_corrected_scorer_rejudge379_postgres_test as prior
from tests.lab_arena import oct01_corrected_scorer_hold378_postgres_test as held


MIGRATION = Path(__file__).parents[2] / 'scripts/381-arena-2026-10-01-attribute-evidence-rejudge.sql'
HOLD = Path(__file__).parents[2] / 'scripts/380-arena-2026-10-01-required-attribute-evidence-hold.sql'
database = prior.database
ROUND = prior.ROUND
NEW_DIGEST = 'sha256:ac4fcea361e079c5ed2dcee87d6cc77eb6bbcfaa2b73a4dd9ea05a7594160f65'


def _render_frozen_hashes(cursor, sql):
    hashes = {
        '40cf904622f4e829d9e3eba89858ff879a6ad0e886738840e85d68fac9479e97':
            held._hash(cursor, "(SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        '872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19':
            held._hash(cursor, "(SELECT participants FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')"),
        '8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61':
            held._hash(cursor, "(SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260930)"),
        '1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3':
            held._hash(cursor, "(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='arena-2026-10-01')"),
        '24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa':
            held._hash(cursor, "(SELECT jsonb_agg(jsonb_build_object('run_id',run_id,'submission_id',submission_id,'stage',stage,'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,'output_ref',output_ref,'stage_generation',stage_generation) ORDER BY icp_position) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='execute')"),
    }
    for expected, actual in hashes.items():
        assert expected in sql
        sql = sql.replace(expected, actual)
    return sql


def _prepare(cursor):
    old_sql, accepted_miner = prior._prepare(cursor)
    cursor.execute(old_sql)
    cursor.execute('SET session_replication_role=replica')
    cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1_scoring',"
                   "status_generation=18,stage_generation=15 WHERE round_id=%s", (ROUND,))
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
        "icp_position,attempt,kind,status,stage_generation,scored_run_id,"
        "terminal_cause,output_ref) "
        "SELECT 'score381:'||e.icp_position||':1',"
        "e.round_id||':'||e.submission_id||':1:'||e.icp_position||':score:rerun379',"
        "e.round_id,e.submission_id,e.miner_hotkey,1,e.icp_position,1,'score',"
        "CASE WHEN e.icp_position<7 THEN 'accepted' ELSE 'pending' END,15,e.run_id,"
        "CASE WHEN e.icp_position<7 THEN 'accepted' ELSE NULL END,"
        "CASE WHEN e.icp_position<7 THEN 'arena/test/score381.json' ELSE NULL END "
        "FROM public.lab_arena_runs e WHERE e.round_id=%s AND e.stage=1 AND e.kind='execute'",
        (ROUND,),
    )
    assert cursor.rowcount == 10
    cursor.execute('SET session_replication_role=origin')
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd) "
        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,1,"
        "'sha256:'||repeat('e',64),'deepline','test_judge','host',100 "
        "FROM public.lab_arena_runs WHERE run_id='score381:0:1'"
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_trajectory_events "
        "(run_id,event_id,round_id,submission_id,miner_hotkey,"
        "runner_hotkey,assignment_id,icp_identifier,stage,icp_position,"
        "attempt,run_kind,model_role,event_kind,occurred_at,content) "
        "SELECT run_id,'00000000-0000-0000-0000-000000000381',round_id,"
        "submission_id,miner_hotkey,miner_hotkey,assignment_id,'oct01-0',"
        "stage,icp_position,attempt,'score','baseline','provider.response',"
        "'2026-10-02T10:00:00Z','{\"operation_id\":\"test_judge\"}'::jsonb "
        "FROM public.lab_arena_runs WHERE run_id='score381:0:1'"
    )
    hold_sql = _render_frozen_hashes(cursor, HOLD.read_text())
    cursor.execute(hold_sql)
    sql = _render_frozen_hashes(cursor, MIGRATION.read_text())
    assert '__NEW_' not in sql
    for function, expected in (
        ('public.lab_arena_open_scoring_v2(text,smallint,jsonb)',
         'b62f5a27c66f2f26a4f8f3384e5a4afa9db7a89641ac1f7ef09f36dc65760c06'),
        ('public.lab_arena_open_stage(text,smallint,jsonb,integer[])',
         'f1b511c4bd0233772d2a06f20e6fa5b6ba3db7d1a2a7d299f18a8932f53f06dd'),
    ):
        cursor.execute("SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),"
                       "'sha256'),'hex')", (function,))
        sql = sql.replace(expected, cursor.fetchone()[0])
    return sql, accepted_miner


def test_rejudge_archives_old_scores_preserves_miner_and_resumes(database, monkeypatch):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, accepted_miner = _prepare(cursor)
            cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                           "WHERE round_id=%s AND stage=2 ORDER BY run_id", (ROUND,))
            miners_before = cursor.fetchall()
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l "
                           "WHERE round_id=%s AND stage=2 ORDER BY entry_id", (ROUND,))
            miner_cost_before = cursor.fetchall()
            cursor.execute("SELECT to_jsonb(e)-'round_id'-'submission_id' "
                           "FROM public.lab_arena_trajectory_events e "
                           "WHERE run_id='score381:0:1'")
            score_event_before = cursor.fetchall()
            cursor.execute("SELECT to_jsonb(l)-'round_id'-'submission_id' "
                           "FROM public.lab_arena_ledger l "
                           "WHERE run_id='score381:0:1'")
            score_cost_before = cursor.fetchall()
            cursor.execute(sql)
            cursor.execute("SELECT status,status_generation,stage_generation,"
                           "configuration_doc->>'scorer_image_digest' "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            assert cursor.fetchone() == ('stage1_closed', 19, 16, NEW_DIGEST)
            cursor.execute("SELECT operator_paused FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone()[0] is False
            cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r "
                           "WHERE round_id=%s AND stage=2 ORDER BY run_id", (ROUND,))
            assert cursor.fetchall() == miners_before
            cursor.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l "
                           "WHERE round_id=%s AND stage=2 ORDER BY entry_id", (ROUND,))
            assert cursor.fetchall() == miner_cost_before
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'", (ROUND,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT count(*),count(*) FILTER (WHERE status='accepted') "
                           "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                           (ROUND+'-r381archive',))
            assert cursor.fetchone() == (10, 7)
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger "
                           "WHERE round_id=%s AND run_id='score381:0:1'",
                           (ROUND+'-r381archive',))
            assert cursor.fetchone()[0] == 1
            cursor.execute("SELECT to_jsonb(e)-'round_id'-'submission_id' "
                           "FROM public.lab_arena_trajectory_events e "
                           "WHERE run_id='score381:0:1'")
            assert cursor.fetchall() == score_event_before
            cursor.execute("SELECT to_jsonb(l)-'round_id'-'submission_id' "
                           "FROM public.lab_arena_ledger l "
                           "WHERE run_id='score381:0:1'")
            assert cursor.fetchall() == score_cost_before
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
    assert all(r['assignment_id'].endswith(':score:rerun381') for r in scores)
    assert all(r['judgment_scope_doc']['scorer_image_digest'] == NEW_DIGEST for r in scores)
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute('SET session_replication_role=replica')
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',output_ref='arena/test/newscore381.json',"
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
            cursor.execute("SELECT status,stage_generation FROM public.lab_arena_runs "
                           "WHERE run_id=%s", (accepted_miner,))
            assert cursor.fetchone() == ('accepted', 13)
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                           "AND stage=2 AND status='pending' AND stage_generation="
                           "(SELECT stage_generation FROM public.lab_arena_rounds WHERE round_id=%s)",
                           (ROUND, ROUND))
            assert cursor.fetchone()[0] == 129


@pytest.mark.parametrize('mutation', [
    "UPDATE public.lab_arena_runs SET status='leased' WHERE run_id='score381:9:1'",
    "UPDATE public.lab_arena_runs SET status='leased' WHERE round_id='arena-2026-10-01' AND stage=2 AND status='pending' AND icp_position=1",
    "UPDATE public.lab_arena_runs SET per_icp_score=0 WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='execute' AND icp_position=0",
    "UPDATE public.lab_arena_runs SET assignment_id='wrong' WHERE run_id='score381:0:1'",
    "UPDATE public.lab_arena_restart_claim_control SET actor_ref='foreign' WHERE singleton",
    "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{scorer_image_digest}','\"wrong\"'::jsonb) WHERE round_id='arena-2026-10-01'",
])
def test_rejudge_rejects_lease_assignment_hold_or_image_mismatch(database, mutation):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, _ = _prepare(cursor)
            cursor.execute('BEGIN')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute(mutation)
            cursor.execute('SET LOCAL session_replication_role=origin')
            with pytest.raises(psycopg.Error):
                cursor.execute(sql)
            cursor.execute('ROLLBACK')
            cursor.execute("SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s",
                           (ROUND+'-r381archive',))
            assert cursor.fetchone()[0] == 0
