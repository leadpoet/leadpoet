"""Real claims and completions overlap without changing frozen round authority."""
import copy
import json
from pathlib import Path

import pytest

from tests.lab_arena import baseline_deadline_transition564_postgres_test as prior

scenario = prior.scenario
SQL = Path(__file__).parents[2] / 'scripts/572-lab-arena-concurrent-baseline-judging.sql'
scalar = prior.scalar


def prepare(scenario, attempt=1, floor=0):
    db, h, original, saved = scenario
    rid, baseline, judge = prior.prepare(scenario, before=True, attempt=max(1, attempt-1))
    pending = copy.deepcopy(judge)
    pending.update(run_id=judge['assignment_id']+':'+str(attempt), attempt=attempt,
        status='pending', terminal_cause=None, result_doc=None, output_ref=None,
        claim_request_id=None, claim_response=None, lease_token_hash=None,
        lease_expires_at=None, scoring_generation_floor=floor)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage1_scoring',scoring_generation_floor=%s WHERE round_id=%s",(floor,rid))
        cur.execute('DELETE FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>=%s', (judge['assignment_id'],attempt))
        cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*',(json.dumps(pending),))
        # Nonzero-floor control uses genuine saved current scoring authority.
        if floor:
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s AND kind='score' AND run_id<>%s",(rid,pending['run_id']))
            scores=cur.fetchall()
            for (score,) in scores:
                score['scoring_generation_floor']=floor
                cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s',(score['run_id'],))
                cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*',(json.dumps(score),))
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s",(pending['run_id'],))
    h.service._invalidate_hot_round()
    return rid, baseline, pending


def publish_recovered_baseline(scenario, pending):
    db,h,_,_=scenario
    rid=h.round_id
    before=h.service.store.list_runs(rid)
    source=h.service.store.get_run(pending['scored_run_id'])
    assert h.status()=='scored' and source['per_icp_score'] is None
    assert source['qualification_doc'] is None
    assert h.service.store.get_run(pending['run_id'])['status']=='accepted'
    # This fixture has genuine accepted empty outputs: its exact measured
    # projection is zero/empty qualification, materialized only by publish.
    assert json.loads(h.objects.get(source['output_ref']))['companies']==[]
    with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
        preserved=prior.fingerprint(cur,rid)
        assert scalar(cur,'public.lab_arena__baseline_final_ready_v1(%s)',(rid,)) is False
        for patch in ("status='published'","promotion_doc='{}'::jsonb","reward_activated_at=now()"):
            cur.execute('SAVEPOINT unmaterialized572')
            with pytest.raises(db[0].Error,match='lab_arena_final_baseline_incomplete'):
                cur.execute('UPDATE public.lab_arena_rounds SET '+patch+' WHERE round_id=%s',(rid,))
            cur.execute('ROLLBACK TO SAVEPOINT unmaterialized572')
    assert h.service.publish(rid)['status']=='ok'
    assert h.status()=='published'
    projected=h.service.store.get_run(source['run_id'])
    assert projected['per_icp_score']==0
    assert projected['qualification_doc']=={'companies':[]}
    after={run['run_id']:run for run in h.service.store.list_runs(rid)}
    assert set(after)=={run['run_id'] for run in before}
    for run in before:
        if run['kind']=='execute' and run['per_icp_score'] is None:
            assert after[run['run_id']]['per_icp_score']==0
            assert after[run['run_id']]['qualification_doc']=={'companies':[]}
            for key in run.keys()-{'per_icp_score','qualification_doc','updated_at'}:
                assert after[run['run_id']][key]==run[key],key
        else:
            assert after[run['run_id']]==run
    with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
        assert prior.fingerprint(cur,rid)==preserved
    ranking=h.service.store.get_round(rid)['publication_doc']['final_ranking']
    assert len(ranking)==2 and all(row['final_score']==0 for row in ranking)


@pytest.mark.parametrize('attempt,floor', [(1,0),(2,0),(20,0),(1,1)])
def test_real_overlap_and_final_authority(scenario, attempt, floor):
    db,h,_,_=scenario
    rid,baseline,pending=prepare(scenario,attempt,floor)
    # Exact old configuration remains readable by the previous strict client.
    frozen=h.service.store.get_round(rid)
    from lab_arena import contracts
    # The source fixture uses fractional offsets for its shifted wall clock;
    # its original wire document is a normal strict-client configuration.
    assert contracts.validate_round_configuration(scenario[2]['configuration_doc']) == scenario[2]['configuration_doc']
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        installed=scalar(cur,'pg_get_functiondef(%s::regprocedure)',('public.lab_arena_open_stage(text,smallint,jsonb,integer[])',))
        if 'lab_arena_concurrent_baseline_judging_572' not in installed:
            assert h.service.open_stage(rid,2)['status']=='stale'  # failing-before witness
        before=prior.fingerprint(cur,rid)
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        assert prior.fingerprint(cur,rid)==before
    runner=h.runner(0)
    early_lease = runner.claim_one() if (attempt,floor)==(1,0) else None
    early_saved = h.service.store.get_run(early_lease['run_id']) if early_lease else None
    assert h.service.advance_round(rid)['status']=='ok'
    assert h.status()=='stage2'
    row=h.service.store.get_round(rid)
    assert row['configuration_doc']==frozen['configuration_doc']
    assert row['stage1_scoring_plan_doc']==frozen['stage1_scoring_plan_doc']
    assert row['benchmark_ref']==frozen['benchmark_ref']
    assert row['finalists']==sorted(p['submission_id'] for p in row['participants'] if not p['is_king'])
    miners=h.service.store.list_runs(rid,stage=2,kind='execute')
    assert len(miners)==10 and len({r['assignment_id'] for r in miners})==10
    assert h.service.open_stage(rid,2)['status']=='stale'
    assert h.service.store.get_run(pending['run_id'])['status']==('leased' if early_lease else 'pending')
    try:
        # Baseline score gets the first eligible slot despite a miner backlog.
        lease=early_lease or runner.claim_one()
        assert lease['run_id']==pending['run_id'] and lease['status']=='leased'
        preserved=h.service.store.get_run(lease['run_id'])
        if early_saved:
            assert preserved==early_saved
            # A miner can claim and finish while the original baseline lease
            # remains active across the round generation change.
            miner_lease=runner.claim_one()
            assert miner_lease['kind']=='execute' and miner_lease['stage']==2
            assert h.service.store.get_run(lease['run_id'])==preserved
            assert runner._slots.acquire(blocking=False)
            runner._run_lease(miner_lease)
            assert h.service.store.get_run(miner_lease['run_id'])['status']=='accepted'
        assert h.service.open_stage(rid,2)['status']=='stale'
        assert h.service.store.get_run(lease['run_id'])==preserved
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            assert scalar(cur,'public.lab_arena__baseline_final_ready_v1(%s)',(rid,)) is False
            for patch in ("status='published'","promotion_doc='{}'::jsonb","reward_activated_at=now()"):
                cur.execute('SAVEPOINT final572')
                with pytest.raises(db[0].Error,match='lab_arena_final_baseline_incomplete'):
                    cur.execute('UPDATE public.lab_arena_rounds SET '+patch+' WHERE round_id=%s',(rid,))
                cur.execute('ROLLBACK TO SAVEPOINT final572')
        assert runner._slots.acquire(blocking=False)
        runner._run_lease(lease)
        assert h.service.store.get_run(lease['run_id'])['status']=='accepted'
    finally:
        runner.close()
    h.advance_until('scored',runners=1)
    publish_recovered_baseline(scenario,pending)


@pytest.mark.parametrize('variant',['missing','pending_source','duplicate','binding','deadline','cancelled','published','legacy'])
def test_admission_bounds(scenario,variant):
    db,h,_,_=scenario
    rid,baseline,pending=prepare(scenario)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
        conn.commit()
        cur.execute('SET LOCAL session_replication_role=replica')
        source=pending['scored_run_id']
        if variant=='missing': cur.execute('DELETE FROM public.lab_arena_runs WHERE run_id=%s',(source,))
        elif variant=='pending_source': cur.execute("UPDATE public.lab_arena_runs SET status='pending' WHERE run_id=%s",(source,))
        elif variant=='binding': cur.execute("UPDATE public.lab_arena_runs SET assignment_id='wrong' WHERE run_id=%s",(source,))
        elif variant=='duplicate':
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',(source,))
            duplicate=cur.fetchone()[0]
            duplicate.update(run_id=duplicate['assignment_id']+':duplicate:1',
                assignment_id=duplicate['assignment_id']+':duplicate',attempt=1,claim_request_id=None)
            cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*',(json.dumps(duplicate),))
        elif variant=='deadline': cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb((now()-interval '1 second')::TEXT)) WHERE round_id=%s",(rid,))
        elif variant=='legacy': cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc-'execution_sequence_policy' WHERE round_id=%s",(rid,))
        else: cur.execute('UPDATE public.lab_arena_rounds SET status=%s WHERE round_id=%s',(variant,rid))
    assert h.service.open_stage(rid,2)['status']=='stale'
    assert h.service.store.list_runs(rid,stage=2,kind='execute')==[]
    if variant in ('missing','pending_source','duplicate','binding'):
        h.service.advance_round(rid)
        assert h.status()!='stage2'
        assert h.service.store.list_runs(rid,stage=2,kind='execute')==[]


def test_miners_finish_while_baseline_retry_keeps_its_backoff(scenario):
    db,h,_,_=scenario
    rid,_,pending=prepare(scenario,attempt=20)
    with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
        cur.execute(SQL.read_text())
        conn.commit()
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_runs SET created_at=now() WHERE run_id=%s',(pending['run_id'],))
    assert h.service.advance_round(rid)['status']=='ok'
    h.run_stage_with_runners(1)
    assert all(r['status']=='accepted' for r in h.service.store.list_runs(rid,stage=2,kind='execute'))
    assert h.service.store.get_run(pending['run_id'])['status']=='pending'
    h.advance_until('stage2_scoring',runners=1)
    h.run_stage_with_runners(1)
    assert h.service.advance_round(rid)['status']=='waiting'
    assert h.status()=='stage2_scoring'
    assert h.service.store.get_run(pending['run_id'])['status']=='pending'
    with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
        assert scalar(cur,'public.lab_arena__baseline_final_ready_v1(%s)',(rid,)) is False
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s",(pending['run_id'],))
    h.run_stage_with_runners(1)
    assert h.service.store.get_run(pending['run_id'])['status']=='accepted'
    h.advance_until('scored',runners=1)
    publish_recovered_baseline(scenario,pending)


@pytest.mark.parametrize('signature',[
    'lab_arena_open_stage(text,smallint,jsonb,integer[])',
    'lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
    'lab_arena__baseline_execution_ready_v1(text)',
])
def test_migration_rejects_source_and_acl_drift(scenario,signature):
    db,_,_,_=scenario
    with db[0].connect(**db[1]) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            cur.execute(SQL.read_text())
            original=scalar(cur,'pg_get_functiondef(%s::regprocedure)',('public.'+signature,))
            cur.execute(original.replace('BEGIN','BEGIN\n  -- injected source drift572',1))
            with pytest.raises(db[0].Error,match='differs'):
                cur.execute(SQL.read_text())
            cur.execute('ROLLBACK')
            cur.execute(original)
            cur.execute('GRANT EXECUTE ON FUNCTION public.'+signature+' TO anon')
            with pytest.raises(db[0].Error):
                cur.execute(SQL.read_text())
            cur.execute('ROLLBACK')
            cur.execute('REVOKE EXECUTE ON FUNCTION public.'+signature+' FROM anon')
            cur.execute(SQL.read_text())
