"""Archived authority restoration uses saved judgments and ordinary publication."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import pytest
from tests.lab_arena import recovery_latest_call_heads537_postgres_test as prior
from tests.lab_arena import evaluation_recovery536_postgres_test as recovery
base=recovery.base
invoke=recovery.invoke
SQL=Path(__file__).parents[2]/'scripts/538-lab-arena-restore-recorded-results.sql'
SIG='public.lab_arena_restore_recorded_results_v1(text,jsonb,bigint,bigint,bigint,text,text,text)'

@pytest.fixture(scope='module')
def database():
    gen=prior.database.__wrapped__();db=next(gen)
    try:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute(SQL.read_text());cur.execute(SQL.read_text())
            cur.execute("""SELECT 'public.'||p.oid::regprocedure::text,jsonb_build_object(
              'sha256',encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex'),
              'source_sha256',encode(extensions.digest(p.prosrc,'sha256'),'hex'),
              'owner',pg_get_userbyid(p.proowner),'acl',p.proacl::text,'definer',p.prosecdef,
              'volatility',p.provolatile,'config',p.proconfig,'language',l.lanname)
              FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.pronamespace='public'::regnamespace
              AND p.proname=ANY(%s)""",(['lab_arena_restore_recorded_results_v1','lab_arena_scoring_policy_history_v1',
                'lab_arena_runs_terminal_v1','lab_arena_integrity_run_guard_v1',
                'lab_arena_operator_hold_transition_guard_v1','lab_arena_operator_hold_score_guard_v1'],))
            catalog=dict(cur.fetchall())
            if os.environ.get('ARENA_FUNCTION_CATALOG_OUT'):
                Path(os.environ['ARENA_FUNCTION_CATALOG_OUT']).write_text(json.dumps(catalog,indent=2)+'\n')
        yield db
    finally:gen.close()

def restore(cur,row,floor,request='restore-original-request-0001'):
    cur.execute('SELECT '+SIG.split('(')[0]+'(%s,%s::jsonb,%s,%s,%s,%s,%s,%s)',
      (row['round_id'],json.dumps(row['configuration_doc']),row['status_generation'],row['stage_generation'],floor,request,'test-operator','model-policy-regrade'))
    return cur.fetchone()[0]


def test_restore_is_owner_only(database):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        for role in ('anon','authenticated','service_role','lab_arena_service'):
            cur.execute('SELECT has_function_privilege(%s,%s,%s)',(role,SIG,'EXECUTE'))
            assert cur.fetchone()==(False,)


def add_original_incomplete_cohorts(database,rid):
    """Only disposable legacy fixture construction bypasses INSERT guards."""
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(rid,));row=cur.fetchone()[0]
        original=next(p for p in row['participants'] if not p.get('is_king'))
        participants=deepcopy(row['participants']);plan=deepcopy(row['stage2_scoring_plan_doc'])
        for sub,positions in [('restore-gap',[0]),('restore-credentials',[0,1])]:
            participant=deepcopy(original);participant.update(submission_id=sub,miner_hotkey=base.company.original.hotkey(sub))
            participants.append(participant)
            cur.execute("INSERT INTO public.lab_arena_submissions SELECT (jsonb_populate_record(NULL::public.lab_arena_submissions,to_jsonb(s)||%s::jsonb)).* FROM public.lab_arena_submissions s WHERE submission_id=%s",(json.dumps({'submission_id':sub,'miner_hotkey':participant['miner_hotkey']}),original['submission_id']))
            for position in positions:
                assignment=f'{rid}:{sub}:2:{position}'
                patch={'run_id':assignment+':1','assignment_id':assignment,'submission_id':sub,'miner_hotkey':participant['miner_hotkey'],
                  'icp_position':position,'status':'failed','terminal_cause':'stage_closed' if sub=='restore-gap' else 'credential_error',
                  'terminal_doc':{'previous_status':'pending','infrastructure_incomplete':True} if sub=='restore-gap' else {},
                  'output_ref':None,'result_doc':None,'per_icp_score':None,'qualification_doc':None,'runner_hotkey':None,
                  'lease_generation':0,'lease_token_hash':None,'lease_expires_at':None,'claim_request_id':None,'claim_response':None}
                cur.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(s)||%s::jsonb)).* FROM public.lab_arena_runs s WHERE round_id=%s AND submission_id=%s AND kind='execute' AND icp_position=%s",(json.dumps(patch),rid,original['submission_id'],position))
                key='incomplete_rows' if sub=='restore-gap' else 'zero_rows'
                plan.setdefault(key,[]).append({'submission_id':sub,'icp_position':position,'reason':patch['terminal_cause']})
        cur.execute('UPDATE public.lab_arena_rounds SET participants=%s::jsonb,stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s',(json.dumps(participants),json.dumps(plan),rid))
        cur.execute('SET LOCAL session_replication_role=origin')

@pytest.mark.parametrize('active_cutoff',[False,True])
def test_archived_authority_restores_through_ordinary_aggregate_and_publication(database,tmp_path,monkeypatch,active_cutoff):
    from dataclasses import replace
    from lab_arena import scoring
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    from tests.lab_arena import test_lab_arena_service_round as flows
    from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge,_enable_verified_proxy_runtime
    from tests.lab_arena.icp_fixtures import daily_icps
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='' WHERE singleton")
        cur.execute('SET LOCAL session_replication_role=origin')
    harness=IntegrityHarness(lambda:database[0].connect(**database[1]),tmp_path,challengers=['Control'],runners=['alpha'])
    harness.service.config.defaults=replace(harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z',company_judgment_from='2000-01-01T00:00:00Z',
        per_icp_cost_policy=True,rewards_enabled=False,benchmark_icp_count=2)
    harness.service.config.daily_icp_source=lambda **kwargs:{'status':'ready','set_id':kwargs['set_id'],'icps':daily_icps()[:2]}
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(flows,'deterministic_scorer',_baseline_first_judge)
    monkeypatch.setitem(scoring.DEFAULT_JUDGE_MODELS,'role_batch_check','google/gemini-2.5-flash')
    original_prices=flows.price_table
    monkeypatch.setattr(flows,'price_table',lambda:original_prices((*flows.PRICED_MODELS,'google/gemini-2.5-flash')))
    # Empty saved-output controls exercise the entire real input/claim/output/
    # score/publication path without inventing external evidence or paid calls.
    harness.sandbox.empty_outputs.update((flavor,position) for flavor in ('Control','PublicBaseline') for position in (0,1))
    try:
        assert flows._start_round(harness,day=22,epoch=62517)==2
        harness.clock.advance_to(harness.schedule()['stage_1_start'])
        harness.advance_until('scored',runners=1)
        # Match a cancelled round whose miner aggregates were not persisted.
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE round_id=%s AND stage=2 AND kind='execute'",(harness.round_id,))
            cur.execute('SET LOCAL session_replication_role=origin')
        if active_cutoff:
            add_original_incomplete_cohorts(database,harness.round_id)
        row=harness.service.store.get_round(harness.round_id)
        old_judges=harness.service.store.list_runs(harness.round_id,kind='score')
        assert len(old_judges)==4 and all(r['status']=='accepted' for r in old_judges)
        saved_executions=harness.service.store.list_runs(harness.round_id,kind='execute')
        new_row=deepcopy(row)
        new_row['configuration_doc']['scorer_image_digest']='sha256:'+'9'*64
        new_row['configuration_doc']['scorer_image_reference']=flows.SCORER_IMAGE_REFERENCE.rsplit('@',1)[0]+'@'+new_row['configuration_doc']['scorer_image_digest']
        now=datetime.now(timezone.utc)
        for key,hours,seconds in [('stage_2_close',2,0),('final_scoring_close',4,0),('publication_deadline',4,1)]:
            # The protected preparer uses ISO8601 UTC offsets, not only Z.
            new_row['configuration_doc']['schedule'][key]=(now+timedelta(hours=hours,seconds=seconds)).isoformat()
        items=[]
        for stage in (1,2):
            items.extend(harness.service._scoring_work_items(new_row,stage,row['stage%d_scoring_plan_doc'%stage]))
        assert len(items)==4 and all(item['company_judgment_refs']==[] for item in items)
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton');generation=cur.fetchone()[0]
            assert base.operator_hold(cur,row,generation,True)['status']=='held'
            revised=invoke(cur,row,new_row['configuration_doc'],items)
            cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id=%s',(row['round_id'],));fresh=cur.fetchone()[0]
            active=None
            if active_cutoff:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,lease_generation=1,lease_token_hash=%s,lease_expires_at=clock_timestamp()+interval '1 hour',claim_request_id='discarded-worker-request' WHERE round_id=%s AND submission_id='restore-gap' AND attempt=2 RETURNING to_jsonb(lab_arena_runs)",(base.company.original.hotkey('runner-discarded'),'a'*64,row['round_id']))
                active=cur.fetchone()[0]
                cur.execute('SET LOCAL session_replication_role=origin')
                for native_kind in ('reservation','dispatch','uncertain'):
                    cur.execute('SAVEPOINT active_native')
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','host',0 FROM public.lab_arena_runs WHERE run_id=%s",(native_kind,'sha256:'+'d'*64,active['run_id']))
                    cur.execute('SET LOCAL session_replication_role=origin')
                    with pytest.raises(database[0].Error):restore(cur,fresh,revised['generation_floor'])
                    cur.execute('ROLLBACK TO SAVEPOINT active_native')
            # Every failure is a real owner RPC call, inside a rollback savepoint.
            for fault in ('stale','published','rewarded','unheld','selected_lease','open_provider'):
                cur.execute('SAVEPOINT rejected_restore')
                cur.execute('SET LOCAL session_replication_role=replica')
                if fault=='published':
                    cur.execute("UPDATE public.lab_arena_rounds SET published_at=clock_timestamp() WHERE round_id=%s",(row['round_id'],))
                elif fault=='rewarded':
                    cur.execute("UPDATE public.lab_arena_rounds SET reward_basis_hash='sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' WHERE round_id=%s",(row['round_id'],))
                elif fault=='unheld':
                    cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton")
                elif fault=='selected_lease':
                    cur.execute("UPDATE public.lab_arena_runs SET status='leased' WHERE run_id=%s",(saved_executions[0]['run_id'],))
                elif fault=='open_provider':
                    cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','host',0 FROM public.lab_arena_runs WHERE run_id=%s",('sha256:'+'e'*64,saved_executions[0]['run_id']))
                cur.execute('SET LOCAL session_replication_role=origin')
                candidate=deepcopy(fresh)
                if fault=='stale':candidate['stage_generation']+=1
                with pytest.raises(database[0].Error):restore(cur,candidate,revised['generation_floor'])
                cur.execute('ROLLBACK TO SAVEPOINT rejected_restore')
                cur.execute("SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE round_id=%s",(row['round_id'],))
                assert cur.fetchone()==(1,)
            result=restore(cur,fresh,revised['generation_floor'])
            assert result['status']=='restored' and result['selected_generation_floor']==0
            assert len(result['retained_participants'])==2
            if active_cutoff:
                assert {p['submission_id'] for p in result['excluded_participants']}=={'restore-gap','restore-credentials'}
                assert all(p['status']=='evaluation_incomplete' for p in result['excluded_participants'])
                cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(active['run_id'],));stopped=cur.fetchone()[0]
                assert stopped['status']=='failed' and stopped['terminal_cause']=='stage_closed'
                assert stopped['terminal_doc']['previous_status']=='leased'
                for key in ('lease_token_hash','runner_hotkey','lease_generation','lease_expires_at','claim_request_id'):
                    assert stopped[key]==active[key]
                cur.execute("SELECT recovery_doc->'stopped_runs' FROM public.lab_arena_scoring_supersessions WHERE round_id=%s AND generation_floor=%s",(row['round_id'],result['revision_generation']))
                assert active in cur.fetchone()[0]
                cur.execute('SAVEPOINT late_result')
                with pytest.raises(database[0].Error):
                    cur.execute('SELECT public.lab_arena__lock_current_lease(%s,%s)',(active['run_id'],active['lease_token_hash']))
                cur.execute('ROLLBACK TO SAVEPOINT late_result')
            assert restore(cur,fresh,revised['generation_floor'])['status']=='existing'
            cur.execute('SAVEPOINT repeated_restore')
            with pytest.raises(database[0].Error):restore(cur,fresh,revised['generation_floor'],request='restore-different-request-0002')
            cur.execute('ROLLBACK TO SAVEPOINT repeated_restore')
            cur.execute('SELECT public.lab_arena_scoring_policy_history_v1(%s)',(row['round_id'],));history=cur.fetchone()[0]
            restoration=history['revisions'][-1]['restoration']
            assert restoration['selected_generation_floor']==0
            assert restoration['excluded_participants']==result['excluded_participants']
            assert 'stopped_runs' not in json.dumps(history) and 'operator_reason' not in json.dumps(history)
            cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id=%s',(row['round_id'],));restored=cur.fetchone()[0]
            assert restored['stage_generation']>fresh['stage_generation']
            assert restored['configuration_doc']==row['configuration_doc']
            assert restored['status']=='stage2_judged'
            assert base.operator_hold(cur,restored,generation,False)['status']=='released'
        harness.clock.now=now+timedelta(hours=2,seconds=1)
        harness.advance_until('published',runners=1)
        final=harness.service.store.get_round(harness.round_id)
        assert final['status']=='published' and final['scoring_generation_floor']==0
        judges=harness.service.store.list_runs(harness.round_id,kind='score')
        assert [r for r in judges if r['attempt']==1]==old_judges
        assert not [r for r in judges if r['scoring_generation_floor']==revised['generation_floor'] and r['status']=='accepted']
        executions=harness.service.store.list_runs(harness.round_id,kind='execute')
        selected={p['submission_id'] for p in final['participants']}
        selected_executions=[r for r in executions if r['submission_id'] in selected]
        assert len(selected_executions)==4
        assert all(r['per_icp_score'] is not None and r['qualification_doc'] is not None for r in selected_executions)
        assert [(r['run_id'],r['output_ref']) for r in executions if r['attempt']==1]==[(r['run_id'],r['output_ref']) for r in saved_executions]
        assert all(r['per_icp_score'] is None for r in executions if r['submission_id'] not in selected)
        for participant in final['participants']:
            public=harness.service.public_results(harness.round_id,participant['submission_id'])
            assert public['submission_scores']['final'] is not None
    finally:harness.service.store.close()


@pytest.mark.parametrize('fault',['grant','source'])
def test_migration_rejects_unreviewed_restore_authority(database,fault):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));original=cur.fetchone()[0]
        if fault=='grant':cur.execute('GRANT EXECUTE ON FUNCTION '+SIG+' TO lab_arena_service')
        else:cur.execute(original.replace('recorded restoration input invalid','unreviewed restoration input'))
        conn.commit()
        try:
            with pytest.raises(database[0].Error):cur.execute(SQL.read_text())
        finally:
            conn.rollback()
            if fault=='grant':cur.execute('REVOKE EXECUTE ON FUNCTION '+SIG+' FROM lab_arena_service')
            else:cur.execute(original)
