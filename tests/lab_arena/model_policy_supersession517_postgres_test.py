"""Exact operator revision keeps historical work and queues uniform new authority."""
from copy import deepcopy
import json
import re
from pathlib import Path
import pytest
from lab_arena import contracts
from lab_arena.store import ArenaStore,PsycopgTransport
from tests.lab_arena import leased_image_setup_recovery515_postgres_test as base
from tests.lab_arena import legacy_company_cache501_postgres_test as company

SQL=Path(__file__).parents[2]/'scripts/517-lab-arena-model-policy-supersession.sql'

def preimage_database():
    gen=base.database.__wrapped__();db=next(gen)
    try:
        with db[0].connect(**db[1]) as conn:
            base.apply(conn)
            with conn.cursor() as cur:
                cur.execute(next((base.ROOT/'scripts').glob('516-*.sql')).read_text())
                # Match the independently retained production security catalog.
                # The historical harness intentionally hardens trigger ACLs and
                # omits the production read-only parity role.
                cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_integrity_run_guard_v1() TO anon,authenticated,service_role')
                cur.execute("DO $$BEGIN IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='leadpoet_parity_reader') THEN CREATE ROLE leadpoet_parity_reader NOLOGIN; END IF; END$$")
                cur.execute('GRANT SELECT ON public.lab_arena_restart_claim_control TO leadpoet_parity_reader')
                for relation in ('lab_arena_rounds','lab_arena_runs','lab_arena_published_results_v1'):
                    cur.execute('REVOKE ALL ON public.'+relation+' FROM lab_arena_service,leadpoet_parity_reader')
                    cur.execute('GRANT SELECT ON public.'+relation+' TO leadpoet_parity_reader,lab_arena_service')
                cur.execute("SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions) FROM pg_class c WHERE c.oid='public.lab_arena_published_results_v1'::regclass")
                assert cur.fetchone()[0]==['lab_arena_owner',[['lab_arena_service','SELECT',False,'lab_arena_owner'],['leadpoet_parity_reader','SELECT',False,'lab_arena_owner']],False,False,['security_invoker=true']]
                # Match postgres's live public-schema defaults. A new private
                # history table must remove these inherited grants explicitly.
                cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO anon,service_role')
                cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT SELECT ON TABLES TO leadpoet_parity_reader')
                cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
        yield db
    finally:gen.close()


@pytest.fixture(scope='module')
def database():
    gen=preimage_database();db=next(gen)
    try:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute(SQL.read_text());cur.execute(SQL.read_text())
        yield db
    finally:gen.close()


def seed(db):
    store=ArenaStore(PsycopgTransport(lambda:db[0].connect(**db[1])))
    store._test_database=db
    rid='arena-2099-10-20-supersession'
    # Only legacy immutable fixture construction uses replica inserts, which
    # predate the trusted INSERT guard. Restore the real NULL default before RPCs.
    with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
        cur.execute('ALTER TABLE public.lab_arena_runs ALTER COLUMN scoring_generation_floor SET DEFAULT 0')
    try:
        participants,items=company.open_new(store,round_id=rid,markers_by_submission=[['base'],['miner']])
    finally:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute('ALTER TABLE public.lab_arena_runs ALTER COLUMN scoring_generation_floor DROP DEFAULT')
    # Real baseline stage1 and miner stage2 use their original execution plans.
    participants[0]['is_king']=True
    miner=items[1]
    old_source=miner['scored_run_id']
    miner['scored_run_id']=old_source.replace(':1:0:',':2:0:')
    with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET run_id=replace(run_id,':1:0:',':2:0:'),assignment_id=replace(assignment_id,':1:0',':2:0'),scored_run_id=replace(scored_run_id,':1:0:',':2:0:'),stage=2 WHERE round_id=%s AND submission_id=%s",(rid,miner['submission_id']))
        cur.execute("UPDATE public.lab_arena_rounds SET participants=%s::jsonb,stage1_scoring_plan_doc=%s::jsonb,stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s",(json.dumps(participants),json.dumps({'work_items':[items[0]]}),json.dumps({'work_items':[items[1]]}),rid))
        cur.execute('UPDATE public.lab_arena_submissions SET is_king=true WHERE submission_id=%s',(participants[0]['submission_id'],))
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged',configuration_doc=jsonb_set(configuration_doc,'{execution_sequence_policy}','\"baseline_scored_first_v1\"') WHERE round_id=%s",(rid,))
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{scorer_policy,judge_models,role_batch_check}','\"google/gemini-2.5-flash\"') WHERE round_id=%s",(rid,))
        cur.execute("UPDATE public.lab_arena_runs SET per_icp_score=42 WHERE round_id=%s AND kind='execute'",(rid,))
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE round_id=%s AND kind='score'",(rid,))
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,actor_ref='test-operator',pause_reason='model-policy-regrade',candidate_commit='',restart_scope='',restart_phase='' WHERE singleton")
        cur.execute('SET LOCAL session_replication_role=origin')
    row=store.get_round(rid)
    new=deepcopy(items)
    for item in new:
        for ref in item['company_judgment_refs']:
            ref['company_input_hash']=ref['scope_doc']['company_input_hash']=contracts.document_hash([ref['company_input_hash'],'google/gemini-3.5-flash-lite'])
            ref['scope_doc'].pop('cache_key')
            ref['cache_key']=ref['scope_doc']['cache_key']=contracts.document_hash(ref['scope_doc'])
    return store,row,items,new


def invoke(cur,row,items,request='revision-request-0001'):
    cur.execute('SELECT public.lab_arena_supersede_role_model_v1(%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s,%s)',
        (row['round_id'],json.dumps(row['configuration_doc']),row['status_generation'],row['stage_generation'],
         'google/gemini-3.5-flash-lite',json.dumps(items),request,'test-operator'))
    return cur.fetchone()[0]


def test_exact_revision_uniform_queue_history_and_idempotence(database):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            result=invoke(cur,row,new)
            assert result['status']=='superseded' and result['queued']==2
            assert invoke(cur,row,new)['status']=='existing'
            cur.execute("SELECT count(*),count(*) FILTER(WHERE per_icp_score IS NULL) FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute'",(row['round_id'],))
            assert cur.fetchone()==(2,2)
            cur.execute("SELECT stage_generation,company_judgment_refs FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND attempt=2",(row['round_id'],))
            queued=cur.fetchall();assert len(queued)==2
            assert all(r[0]==result['generation_floor'] for r in queued)
            cur.execute('SELECT old_round_doc,new_round_doc,old_score_projections FROM public.lab_arena_scoring_supersessions WHERE round_id=%s',(row['round_id'],))
            before,after,projections=cur.fetchone()
            assert before['configuration_doc']==row['configuration_doc']
            expected=deepcopy(before['configuration_doc'])
            expected['scorer_policy']['judge_models']['role_batch_check']='google/gemini-3.5-flash-lite'
            expected['scorer_policy']['env_bindings']['ARENA_ROLE_BATCH_MODEL']='google/gemini-3.5-flash-lite'
            assert after['configuration_doc']==expected
            assert all(p['per_icp_score']==42 for p in projections.values())
            cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND attempt=1 AND stage_generation<%s",(row['round_id'],result['generation_floor']))
            assert cur.fetchone()==(2,)
            cur.execute("SELECT public.lab_arena__judge_continuation_v1(a,r) FROM public.lab_arena_runs r JOIN public.lab_arena_rounds a USING(round_id) WHERE r.round_id=%s AND r.kind='score' ORDER BY r.attempt",(row['round_id'],))
            decisions=[v[0] for v in cur.fetchall()]
            assert decisions==[False,False,True,True]
            cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)',(row['round_id'],))
            assert cur.fetchone()==(False,)
            cur.execute('SELECT public.lab_arena_scoring_policy_history_v1(%s)',(row['round_id'],))
            history=cur.fetchone()[0]
            assert history=={'round_id':row['round_id'],'revisions':[{
                'previous_generation_floor':0,'generation_floor':result['generation_floor'],
                'old_scorer_policy':before['configuration_doc']['scorer_policy'],
                'new_scorer_policy':expected['scorer_policy']}]}
        assert store.scoring_policy_history(row['round_id'])==history
    finally:store.close()

@pytest.mark.parametrize('fault',['unpaused','stale_config','stale_generation','missing_work','duplicate_work','old_cache','leased','pending','published','open_charge'])
def test_revision_rejects_unsafe_or_mixed_inputs_without_changes(database,fault):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            if fault in ('unpaused','leased','pending','published'):
                cur.execute('SET LOCAL session_replication_role=replica')
                if fault=='unpaused':cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
                if fault=='leased':cur.execute("UPDATE public.lab_arena_runs SET status='leased' WHERE run_id=%s",(old[0]['scored_run_id'],))
                if fault=='pending':cur.execute("UPDATE public.lab_arena_runs SET status='pending',terminal_cause=NULL WHERE run_id=%s",(old[1]['scored_run_id'][:-2]+':score:1',))
                if fault=='published':cur.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s",(row['round_id'],))
                cur.execute('SET LOCAL session_replication_role=origin')
            if fault=='open_charge':
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','miner_key',0 FROM public.lab_arena_runs WHERE run_id=%s",('sha256:'+'d'*64,old[1]['scored_run_id']))
            if fault=='stale_config':row['configuration_doc']['untrusted']='changed'
            if fault=='stale_generation':row['stage_generation']+=1
            if fault=='missing_work':new=new[:-1]
            if fault=='duplicate_work':new=[new[0],new[0]]
            if fault=='old_cache':new=old
            cur.execute('SAVEPOINT rejected')
            with pytest.raises(database[0].Error):invoke(cur,row,new)
            cur.execute('ROLLBACK TO SAVEPOINT rejected')
            cur.execute('SELECT count(*) FROM public.lab_arena_scoring_supersessions')
            assert cur.fetchone()==(0,)
            cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND attempt>1",(row['round_id'],))
            assert cur.fetchone()==(0,)
    finally:store.close()


def test_untrusted_roles_cannot_authorize_history_or_call_operator(database):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        for role in ('anon','authenticated','service_role','lab_arena_service'):
            cur.execute("SELECT has_function_privilege(%s,'public.lab_arena_supersede_role_model_v1(text,jsonb,bigint,bigint,text,jsonb,text,text)','EXECUTE'),has_table_privilege(%s,'public.lab_arena_scoring_supersessions','INSERT')",(role,role))
            assert cur.fetchone()==(False,False)
            cur.execute("SELECT has_function_privilege(%s,'public.lab_arena_scoring_policy_history_v1(text)','EXECUTE')",(role,))
            assert cur.fetchone()==(role=='lab_arena_service',)
            cur.execute("SELECT has_function_privilege(%s,'public.lab_arena_scoring_operator_hold_v1(text,jsonb,bigint,bigint,bigint,text,text,boolean)','EXECUTE')",(role,))
            assert cur.fetchone()==(False,)

@pytest.mark.parametrize('statement',[
    'GRANT EXECUTE ON FUNCTION public.lab_arena__all_judgments_ready_v1(text) TO anon',
    'ALTER FUNCTION public.lab_arena_runs_terminal_v1() OWNER TO postgres',
    'GRANT DELETE ON public.lab_arena_published_results_v1 TO lab_arena_service',
    'ALTER TABLE public.lab_arena_rounds DISABLE ROW LEVEL SECURITY',
    'GRANT INSERT ON public.lab_arena_scoring_supersessions TO lab_arena_service',
    'GRANT EXECUTE ON FUNCTION public.lab_arena_supersede_role_model_v1(text,jsonb,bigint,bigint,text,jsonb,text,text) TO anon',
    'GRANT EXECUTE ON FUNCTION public.lab_arena_scoring_policy_history_v1(text) TO anon',
    'CREATE POLICY unexpected_history ON public.lab_arena_scoring_supersessions FOR SELECT TO lab_arena_service USING(true)',
    'ALTER TABLE public.lab_arena_scoring_supersessions DISABLE TRIGGER lab_arena_scoring_supersessions_append_only',
    'GRANT EXECUTE ON FUNCTION public.lab_arena_scoring_operator_hold_v1(text,jsonb,bigint,bigint,bigint,text,text,boolean) TO anon',
])
def test_already_installed_migration_rejects_security_drift(database,statement):
    with database[0].connect(**database[1]) as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
            with pytest.raises(database[0].Error,match='security differs'):
                cur.execute(SQL.read_text())
        conn.rollback()


def test_operator_close_pending_then_supersede_atomically_preserves_completed_work(database):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            baseline_score=old[0]['scored_run_id'][:-2]+':score:1'
            miner_score=old[1]['scored_run_id'][:-2]+':score:1'
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='saved-baseline-judgment' WHERE run_id=%s",(baseline_score,))
            cur.execute("UPDATE public.lab_arena_runs SET status='pending',terminal_cause=NULL WHERE run_id=%s",(miner_score,))
            cur.execute('SET LOCAL session_replication_role=origin')
            for kind,amount,response in [('reservation',0,None),('settlement',123,{'status':200,'call_succeeded':True})]:
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,terminal_response) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','host',%s,%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s",(kind,'sha256:'+'f'*64,amount,json.dumps(response),baseline_score))
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',(baseline_score,));saved=cur.fetchone()[0]
            cur.execute('SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l');charges=cur.fetchone()[0]
            cur.execute('SELECT public.lab_arena_close_scoring(%s,2::smallint)',(row['round_id'],))
            closed=cur.fetchone()[0];assert closed['status']=='closed'
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(row['round_id'],));fresh=cur.fetchone()[0]
            assert fresh['stage_generation']==row['stage_generation']+1
            assert fresh['status_generation']==row['status_generation']+1
            assert fresh['status']=='stage2_judged'
            revised=invoke(cur,fresh,new)
            assert revised['queued']==2
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',(baseline_score,));assert cur.fetchone()[0]==saved
            cur.execute('SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l');assert cur.fetchone()[0]==charges
            cur.execute('SELECT status,terminal_cause FROM public.lab_arena_runs WHERE run_id=%s',(miner_score,));assert cur.fetchone()==('failed','stage_closed')
    finally:store.close()

@pytest.mark.parametrize('case',['unpaused','active','no_pending','baseline_only'])
def test_operator_close_does_not_invalidate_other_work(database,case):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if case!='no_pending':
                item=old[0] if case=='baseline_only' else old[1]
                cur.execute("UPDATE public.lab_arena_runs SET status=%s,terminal_cause=NULL WHERE run_id=%s",('leased' if case=='active' else 'pending',item['scored_run_id'][:-2]+':score:1'))
            if case=='unpaused':cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s',(row['round_id'],));saved=cur.fetchone()[0]
            cur.execute('SELECT public.lab_arena_close_scoring(%s,2::smallint)',(row['round_id'],));assert cur.fetchone()[0]['status']=='existing'
            cur.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s',(row['round_id'],));assert cur.fetchone()[0]==saved
            cur.execute('SELECT status_generation,stage_generation FROM public.lab_arena_rounds WHERE round_id=%s',(row['round_id'],));assert cur.fetchone()==(row['status_generation'],row['stage_generation'])
    finally:store.close()


def test_new_roots_claim_and_complete_through_ordinary_company_protocol(database):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            revised=invoke(cur,row,new)
            # Preserve production retry backoff; advance only disposable queue time.
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
        runner=company.original.hotkey('supersession-current-runner')
        for _ in range(2):
            lease,token,_,_=company.original.claim(store,row['round_id'],runner,excluded=[runner])
            assert lease['status']=='leased',lease
            assert lease['attempt']==2
            run=store.get_run(lease['run_id'])
            assert run['stage_generation']==revised['generation_floor']
            assert run['scoring_generation_floor']==revised['generation_floor']
            refs=next(item['company_judgment_refs'] for item in new if item['scored_run_id']==lease['scored_run_id'])
            assert len(lease['company_judgment_cache']['misses'])==len(refs)
            assert company.complete_subset(store,lease,token,refs)['company_judgments_stored']==len(refs)
            assert store.get_run(lease['run_id'])['status']=='accepted'
        scores=store.list_runs(row['round_id'],kind='score')
        assert len([r for r in scores if r['stage_generation']==revised['generation_floor'] and r['status']=='accepted'])==2
        assert len([r for r in scores if r['stage_generation']<revised['generation_floor'] and r['status']=='failed'])==2
    finally:store.close()


def test_ordinary_floor_zero_round_still_scores_and_publishes(database,tmp_path,monkeypatch):
    from tests.lab_arena.claim_admission_transition509_510_postgres_test import test_latest_schema_retains_complete_normal_round_aggregate_and_publication
    test_latest_schema_retains_complete_normal_round_aggregate_and_publication(database,tmp_path,monkeypatch)

@pytest.mark.parametrize('proof,funding,expected',[(True,'miner_key',True),(False,'miner_key',False),(True,'host',False)])
def test_only_native_same_payer_credit_proof_survives_model_revision(database,proof,funding,expected):
    from tests.lab_arena import miner_credit_terminal484_postgres_test as credit
    store,row,old,new=seed(database)
    try:
        run=old[1]['scored_run_id'][:-2]+':score:1'
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='credential_error',result_doc='{\"terminal_status\":\"credential_error\"}'::jsonb WHERE run_id=%s",(run,))
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}','\"2000-01-01T00:00:00Z\"') WHERE round_id=%s",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            credit.credit(conn,run,proof=proof,funding=funding)
            cur.execute('SELECT public.lab_arena_miner_credit_failure_v1(%s)',(run,));assert cur.fetchone()==(expected,)
            cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id=%s',(row['round_id'],));fresh=cur.fetchone()[0]
            cur.execute('SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l');charges=cur.fetchone()[0]
            invoke(cur,fresh,new)
            cur.execute("SELECT r.attempt,public.lab_arena_miner_credit_failure_v1(r.run_id),public.lab_arena__judge_continuation_v1(a,r) FROM public.lab_arena_runs r JOIN public.lab_arena_rounds a USING(round_id) WHERE r.assignment_id=%s ORDER BY attempt",(run[:-2],))
            assert cur.fetchall()==[(1,expected,False),(2,expected,not expected)]
            cur.execute('SELECT public.lab_arena_expire_leases(%s)',(row['round_id'],))
            cur.execute('SELECT status,terminal_cause FROM public.lab_arena_runs WHERE run_id=%s',(run[:-1]+'2',))
            assert cur.fetchone()==(('failed','stage_closed') if expected else ('pending',None))
            cur.execute('SELECT public.lab_arena_miner_credit_failure_v1(%s)',(run[:-1]+'2',));assert cur.fetchone()==(expected,)
            cur.execute('SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l');assert cur.fetchone()[0]==charges
    finally:store.close()


def test_completion_lease_guard_rejects_prior_policy_even_with_valid_token(database):
    from lab_arena.store import hash_lease_token
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            invoke(cur,row,new)
            old_run=old[1]['scored_run_id'][:-2]+':score:1'
            token=hash_lease_token('test-previous-policy-token')
            # Simulate a delayed valid old lease independently of the drain
            # precondition. The active-attempt unique index stays in force.
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE assignment_id=%s AND attempt=2",(old_run[:-2],))
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,lease_token_hash=%s,lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=%s",(token,old_run))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SAVEPOINT stale_completion')
            with pytest.raises(database[0].Error,match='lab_arena_lease_stale'):
                cur.execute('SELECT public.lab_arena__lock_current_lease(%s,%s)',(old_run,token))
            cur.execute('ROLLBACK TO SAVEPOINT stale_completion')
            # Same token, source and lease would be valid without the floor.
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET scoring_generation_floor=0 WHERE round_id=%s',(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SELECT (public.lab_arena__lock_current_lease(%s,%s)).run_id',(old_run,token))
            assert cur.fetchone()==(old_run,)
    finally:store.close()


def test_stale_service_cannot_persist_historical_model_score(database):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='old-accepted-score' WHERE round_id=%s AND kind='score'",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            invoke(cur,row,new)
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
            cur.execute('SELECT public.lab_arena_close_scoring(%s,2::smallint)',(row['round_id'],));assert cur.fetchone()[0]['status']=='closed'
            payload=json.dumps([{'run_id':old[1]['scored_run_id'],'per_icp_score':42,'qualification_doc':{'companies':[]}}])
            cur.execute('SAVEPOINT stale_score')
            with pytest.raises(database[0].Error,match='lab_arena_superseded_score_authority'):
                cur.execute('SELECT public.lab_arena_record_run_scores(%s,2::smallint,%s::jsonb)',(row['round_id'],payload))
            cur.execute('ROLLBACK TO SAVEPOINT stale_score')
            cur.execute('SELECT per_icp_score FROM public.lab_arena_runs WHERE run_id=%s',(old[1]['scored_run_id'],));assert cur.fetchone()==(None,)
            cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)',(row['round_id'],));assert cur.fetchone()==(False,)
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET scoring_generation_floor=0 WHERE round_id=%s',(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SELECT public.lab_arena_record_run_scores(%s,2::smallint,%s::jsonb)',(row['round_id'],payload));assert cur.fetchone()[0]['status']=='ok'
    finally:store.close()


def test_current_policy_retry_keeps_completed_company_authority(database):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            revised=invoke(cur,row,new)
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
        runner=company.original.hotkey('supersession-retry-runner')
        first,token,_,_=company.original.claim(store,row['round_id'],runner,excluded=[runner])
        assert first['status']=='leased'
        refs=next(i['company_judgment_refs'] for i in new if i['scored_run_id']==first['scored_run_id'])
        assert company.complete_subset(store,first,token,refs,terminal='judge_error')['company_judgments_stored']==1
        retained=store.get_company_judgment(refs[0]['cache_key'],0)
        predecessor=store.get_run(first['run_id'])
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        saw_reuse=False
        for _ in range(2):
            lease,token,_,_=company.original.claim(store,row['round_id'],runner,excluded=[runner])
            assert lease['status']=='leased',lease
            current=next(i['company_judgment_refs'] for i in new if i['scored_run_id']==lease['scored_run_id'])
            if lease['scored_run_id']==first['scored_run_id']:
                assert len(lease['company_judgment_cache']['hits'])==1
                assert lease['attempt']==3
                saw_reuse=True
            company.complete_subset(store,lease,token,current)
            assert store.get_run(lease['run_id'])['status']=='accepted'
            assert store.get_run(lease['run_id'])['stage_generation']==revised['generation_floor']
            assert store.get_run(lease['run_id'])['scoring_generation_floor']==revised['generation_floor']
        assert saw_reuse
        assert store.get_company_judgment(refs[0]['cache_key'],0)==retained
        assert store.get_run(first['run_id'])==predecessor
    finally:store.close()


def test_old_accepted_parent_new_roots_complete_and_survive_later_close(database):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1])as conn,conn.cursor()as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='retained-old-judgment' WHERE round_id=%s AND kind='score'",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute("SELECT jsonb_agg(to_jsonb(a)ORDER BY run_id)FROM public.lab_arena_runs a WHERE round_id=%s AND kind='score'",(row['round_id'],));old_rows=cur.fetchone()[0]
            revised=invoke(cur,row,new)
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
        runner=company.original.hotkey('independent-supersession-runner')
        for _ in range(2):
            lease,token,_,_=company.original.claim(store,row['round_id'],runner,excluded=[runner])
            assert lease['status']=='leased'and lease['attempt']==2
            refs=next(x['company_judgment_refs']for x in new if x['scored_run_id']==lease['scored_run_id'])
            assert company.complete_subset(store,lease,token,refs)['company_judgments_stored']==len(refs)
            assert store.get_run(lease['run_id'])['status']=='accepted'
        before=store.get_round(row['round_id']);closed=store.close_scoring(row['round_id'],2);assert closed['status']=='closed'
        after=store.get_round(row['round_id']);assert after['stage_generation']==before['stage_generation']+1
        assert after['scoring_generation_floor']==revised['generation_floor']
        with database[0].connect(**database[1])as conn,conn.cursor()as cur:
            cur.execute("SELECT jsonb_agg(to_jsonb(a)ORDER BY run_id)FROM public.lab_arena_runs a WHERE round_id=%s AND kind='score'AND attempt=1",(row['round_id'],));assert cur.fetchone()[0]==old_rows
            cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)',(row['round_id'],));assert cur.fetchone()==(False,)  # Cleared execution projections still await assembly.
            cur.execute("SELECT count(*) FROM public.lab_arena_runs a WHERE round_id=%s AND kind='score'AND status='accepted'AND stage_generation>=%s",(row['round_id'],revised['generation_floor']));assert cur.fetchone()==(2,)
    finally:store.close()


def clone_run(cur,source,patch):
    cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(r)||%s::jsonb)).* FROM public.lab_arena_runs r WHERE run_id=%s RETURNING run_id,scoring_generation_floor',
                (json.dumps(patch),source))
    return cur.fetchone()


@pytest.mark.parametrize('kind,supplied',[('score',0),('score',9999),('execute',1)])
def test_insert_cannot_choose_revision_key(database,kind,supplied):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            invoke(cur,row,new)
            source=(old[1]['scored_run_id'][:-2]+':score:2') if kind=='score' else old[1]['scored_run_id']
            assignment=source.rsplit(':',1)[0]
            cur.execute('SAVEPOINT supplied_floor')
            with pytest.raises(database[0].Error,match='lab_arena_score_revision_invalid'):
                clone_run(cur,source,{'run_id':assignment+':3','attempt':3,'status':'failed',
                    'terminal_cause':'judge_error','scoring_generation_floor':supplied})
            cur.execute('ROLLBACK TO SAVEPOINT supplied_floor')
    finally:store.close()


def test_revision_stamp_is_immutable_and_accepted_unique_within_policy(database):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            revised=invoke(cur,row,new)
            source=old[1]['scored_run_id'][:-2]+':score:2'
            assignment=source.rsplit(':',1)[0]
            run,floor=clone_run(cur,source,{'run_id':assignment+':3','attempt':3,'status':'failed',
                'terminal_cause':'judge_error','scoring_generation_floor':None})
            assert floor==revised['generation_floor']
            cur.execute('SAVEPOINT immutable_floor')
            with pytest.raises(database[0].Error,match='lab_arena_score_revision_immutable'):
                cur.execute('UPDATE public.lab_arena_runs SET scoring_generation_floor=0 WHERE run_id=%s',(run,))
            cur.execute('ROLLBACK TO SAVEPOINT immutable_floor')
            # Accepted uniqueness uses the stable policy key, even when an
            # ordinary close changed the stage generation of a later attempt.
            accepted={'run_id':assignment+':4','attempt':4,'status':'accepted','terminal_cause':'accepted',
                'output_ref':'current-accepted','stage_generation':revised['generation_floor']+1,
                'scoring_generation_floor':None}
            clone_run(cur,source,accepted)
            cur.execute('SAVEPOINT duplicate_policy')
            with pytest.raises(database[0].Error,match='lab_arena_runs_one_accepted_uq'):
                clone_run(cur,source,{**accepted,'run_id':assignment+':5','attempt':5,
                    'stage_generation':revised['generation_floor']+2})
            cur.execute('ROLLBACK TO SAVEPOINT duplicate_policy')
            execution=old[1]['scored_run_id']
            cur.execute('SAVEPOINT duplicate_execute')
            with pytest.raises(database[0].Error,match='lab_arena_runs_one_accepted_uq'):
                clone_run(cur,execution,{'run_id':execution[:-1]+'2','attempt':2,'scoring_generation_floor':None})
            cur.execute('ROLLBACK TO SAVEPOINT duplicate_execute')
    finally:store.close()


def operator_hold(cur,row,guard_generation,hold,*,actor='test-operator',reason='model-policy-regrade'):
    cur.execute('SELECT public.lab_arena_scoring_operator_hold_v1(%s,%s::jsonb,%s,%s,%s,%s,%s,%s)',
        (row['round_id'],json.dumps(row['configuration_doc']),row['status_generation'],
         row['stage_generation'],guard_generation,actor,reason,hold))
    return cur.fetchone()[0]


def test_saved_outputs_regrade_through_service_persistence_and_publication(database,tmp_path,monkeypatch):
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
        row=harness.service.store.get_round(harness.round_id)
        old_judges=harness.service.store.list_runs(harness.round_id,kind='score')
        assert len(old_judges)==4 and all(r['status']=='accepted' for r in old_judges)
        saved_executions=harness.service.store.list_runs(harness.round_id,kind='execute')
        new_row=deepcopy(row)
        new_row['configuration_doc']['scorer_policy']['judge_models']['role_batch_check']='google/gemini-3.5-flash-lite'
        new_row['configuration_doc']['scorer_policy']['env_bindings']['ARENA_ROLE_BATCH_MODEL']='google/gemini-3.5-flash-lite'
        items=[]
        for stage in (1,2):
            items.extend(harness.service._scoring_work_items(new_row,stage,row['stage%d_scoring_plan_doc'%stage]))
        assert len(items)==4 and all(item['company_judgment_refs']==[] for item in items)
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton');generation=cur.fetchone()[0]
            assert operator_hold(cur,row,generation,True)['status']=='held'
            revised=invoke(cur,row,items)
            cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id=%s',(row['round_id'],));fresh=cur.fetchone()[0]
            assert operator_hold(cur,fresh,generation,False)['status']=='released'
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        harness.advance_until('published',runners=1)
        final=harness.service.store.get_round(harness.round_id)
        assert final['status']=='published' and final['scoring_generation_floor']==revised['generation_floor']
        judges=harness.service.store.list_runs(harness.round_id,kind='score')
        assert [r for r in judges if r['attempt']==1]==old_judges
        assert len([r for r in judges if r['stage_generation']>=revised['generation_floor'] and r['status']=='accepted'])==4
        executions=harness.service.store.list_runs(harness.round_id,kind='execute')
        assert len(executions)==len(saved_executions)==4
        assert all(r['per_icp_score'] is not None and r['qualification_doc'] is not None for r in executions)
        assert [(r['run_id'],r['output_ref']) for r in executions]==[(r['run_id'],r['output_ref']) for r in saved_executions]
        for participant in final['participants']:
            public=harness.service.public_results(harness.round_id,participant['submission_id'])
            assert public['submission_scores']['final'] is not None
    finally:harness.service.store.close()


def test_owner_hold_release_is_idempotent_and_preserves_active_work(database):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
            cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton');before=cur.fetchone()[0]
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL WHERE run_id=%s",(old[1]['scored_run_id'][:-2]+':score:1',))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');runs=cur.fetchone()[0]
            assert operator_hold(cur,row,before['guard_generation'],True)['status']=='held'
            assert operator_hold(cur,row,before['guard_generation'],True)['status']=='existing'
            assert operator_hold(cur,row,before['guard_generation'],False)['status']=='released'
            assert operator_hold(cur,row,before['guard_generation'],False)['status']=='existing'
            cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton');after=cur.fetchone()[0]
            for field in ('operator_paused','pause_reason','actor_ref','updated_at'):before.pop(field);after.pop(field)
            assert before==after
            cur.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r');assert cur.fetchone()[0]==runs
    finally:store.close()


@pytest.mark.parametrize('case',['stale_guard','stale_round','foreign_hold','canonical_guard','foreign_release'])
def test_owner_hold_cas_never_replaces_foreign_control(database,case):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton');generation=cur.fetchone()[0]
            if case=='stale_guard':generation+=1
            if case=='stale_round':row['stage_generation']+=1
            if case in ('foreign_hold','foreign_release'):
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET actor_ref='another-owner' WHERE singleton")
            if case=='canonical_guard':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('a',64),owner_commitment='sha256:'||repeat('b',64),guard_generation=greatest(guard_generation,1),guard_expires_at=clock_timestamp()+interval '1 hour',candidate_commit=repeat('a',40),restart_scope='gateway',restart_phase='draining' WHERE singleton")
                cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton');generation=cur.fetchone()[0]
            cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton');before=cur.fetchone()[0]
            cur.execute('SAVEPOINT rejected_hold')
            with pytest.raises(database[0].Error,match='lab_arena_operator_hold_'):
                operator_hold(cur,row,generation,case!='foreign_release')
            cur.execute('ROLLBACK TO SAVEPOINT rejected_hold')
            cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton');assert cur.fetchone()[0]==before
            conn.rollback()
    finally:store.close()


@pytest.mark.parametrize('bad_floor',['missing',None,True,'current',-1,9999])
def test_score_batch_floor_cas_is_strict_and_atomic(database,bad_floor):
    store,row,old,new=seed(database)
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            revised=invoke(cur,row,new);floor=revised['generation_floor']
            source=old[1]['scored_run_id'];judge=source[:-2]+':score:2'
            # Build otherwise valid current-authority DB controls. Semantic
            # artifacts are exercised by the separate real-service test.
            clone_run(cur,judge,{'run_id':source[:-2]+':score:3','attempt':3,'status':'accepted',
                'terminal_cause':'accepted','output_ref':'current-judge','scoring_generation_floor':None})
            other_source=source.replace(':2:0:',':2:1:')
            other_assignment=other_source[:-2]
            clone_run(cur,source,{'run_id':other_source,'assignment_id':other_assignment,
                'icp_position':1,'scoring_generation_floor':None})
            clone_run(cur,judge,{'run_id':other_assignment+':score:2','assignment_id':other_assignment+':score',
                'scored_run_id':other_source,'icp_position':1,'status':'accepted',
                'terminal_cause':'accepted','output_ref':'other-current-judge','scoring_generation_floor':None})
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
            cur.execute('SELECT public.lab_arena_close_scoring(%s,2::smallint)',(row['round_id'],));assert cur.fetchone()[0]['status']=='closed'
            good={'run_id':source,'per_icp_score':0,'qualification_doc':{'companies':[]},'scoring_generation_floor':floor}
            bad={**good,'run_id':other_source}
            if bad_floor=='missing':bad.pop('scoring_generation_floor')
            else:bad['scoring_generation_floor']=str(floor) if bad_floor=='current' else bad_floor
            cur.execute('SAVEPOINT mixed_floor')
            with pytest.raises(database[0].Error,match='lab_arena_superseded_score_authority'):
                cur.execute('SELECT public.lab_arena_record_run_scores(%s,2::smallint,%s::jsonb)',(row['round_id'],json.dumps([good,bad])))
            cur.execute('ROLLBACK TO SAVEPOINT mixed_floor')
            cur.execute('SELECT per_icp_score FROM public.lab_arena_runs WHERE run_id=ANY(%s) ORDER BY run_id',([source,other_source],));assert cur.fetchall()==[(None,),(None,)]
            correct=[good,{**bad,'scoring_generation_floor':floor}]
            cur.execute('SELECT public.lab_arena_record_run_scores(%s,2::smallint,%s::jsonb)',(row['round_id'],json.dumps(correct)));assert cur.fetchone()[0]['recorded']==2
    finally:store.close()


def test_private_revision_objects_remove_inherited_public_schema_grants(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SELECT jsonb_agg(jsonb_build_array(defaclrole,defaclnamespace,defaclobjtype,defaclacl::TEXT) ORDER BY oid) FROM pg_default_acl")
        defaults_before = cur.fetchone()[0]
        cur.execute(SQL.read_text())
        cur.execute("SELECT jsonb_agg(jsonb_build_array(defaclrole,defaclnamespace,defaclobjtype,defaclacl::TEXT) ORDER BY oid) FROM pg_default_acl")
        assert cur.fetchone()[0] == defaults_before
        cur.execute("SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor))) FROM pg_class c CROSS JOIN LATERAL aclexplode(c.relacl) a WHERE c.oid='public.lab_arena_scoring_supersessions'::regclass AND a.grantee<>c.relowner")
        assert cur.fetchone()[0] == [['lab_arena_service','SELECT',False,'lab_arena_owner']]
        for role in ('anon','authenticated','service_role','leadpoet_parity_reader','lab_arena_service'):
            cur.execute("SELECT has_function_privilege(%s,'public.lab_arena_supersede_role_model_v1(text,jsonb,bigint,bigint,text,jsonb,text,text)','EXECUTE'),has_function_privilege(%s,'public.lab_arena_scoring_operator_hold_v1(text,jsonb,bigint,bigint,bigint,text,text,boolean)','EXECUTE'),has_function_privilege(%s,'public.lab_arena_scoring_policy_history_v1(text)','EXECUTE')",(role,role,role))
            assert cur.fetchone() == (False,False,role=='lab_arena_service')
        # The grant is still a live creator default and still applies to
        # existing audited relations; only this new private table loses it.
        cur.execute("SELECT EXISTS(SELECT 1 FROM pg_default_acl d CROSS JOIN LATERAL aclexplode(d.defaclacl)a WHERE d.defaclrole='postgres'::regrole AND d.defaclnamespace='public'::regnamespace AND d.defaclobjtype='r' AND a.grantee='leadpoet_parity_reader'::regrole AND a.privilege_type='SELECT'),has_table_privilege('leadpoet_parity_reader','public.lab_arena_restart_claim_control','SELECT')")
        assert cur.fetchone() == (True,True)


@pytest.mark.parametrize('owner_already_has_create',[False,True])
def test_non_superuser_postgres_restores_exact_schema_permissions(owner_already_has_create):
    gen=preimage_database();db=next(gen);admin=None
    try:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute('CREATE ROLE schema517_fixture_admin LOGIN SUPERUSER PASSWORD %s',(db[1]['password'],))
            cur.execute('GRANT lab_arena_owner TO postgres')
            cur.execute('ALTER SCHEMA public OWNER TO pg_database_owner')
            cur.execute('REVOKE CREATE ON SCHEMA public FROM PUBLIC,lab_arena_owner')
            if owner_already_has_create:
                cur.execute('GRANT CREATE ON SCHEMA public TO lab_arena_owner')
        admin=db[0].connect(**dict(db[1],user='schema517_fixture_admin'))
        admin.autocommit=True
        with admin.cursor() as cur:
            cur.execute('ALTER ROLE postgres NOSUPERUSER INHERIT CREATEROLE BYPASSRLS')
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute("SELECT current_user,(SELECT rolsuper FROM pg_roles WHERE rolname=current_user),has_schema_privilege('postgres','public','CREATE'),has_schema_privilege('lab_arena_owner','public','CREATE')")
            assert cur.fetchone()==('postgres',False,True,owner_already_has_create)
            cur.execute("SELECT jsonb_build_array(oid,nspowner,nspacl::TEXT) FROM pg_namespace WHERE nspname='public'")
            before=cur.fetchone()[0]
            cur.execute("SELECT jsonb_agg(jsonb_build_array(defaclrole,defaclnamespace,defaclobjtype,defaclacl::TEXT) ORDER BY oid) FROM pg_default_acl")
            defaults=cur.fetchone()[0]
            conn.commit()
            sql=SQL.read_text()
            if not owner_already_has_create:
                original=re.sub(r'DO \$schema_(?:create|restore)517\$.*?\$schema_(?:create|restore)517\$;','',sql,flags=re.S)
                assert original!=sql
                with pytest.raises(db[0].Error,match='permission denied for schema public') as error:
                    cur.execute(original)
                assert error.value.pgcode=='42501'
                conn.rollback()
            # Fail after the temporary permission/new-object creation. All DDL
            # and the permission grant must roll back together.
            broken=sql.replace("'31e88aa868a9caecaffd4571a294e5fb91ce2e29b29748f2157eb98eca05b64b'","'"+'0'*64+"'",1)
            assert broken!=sql
            with pytest.raises(db[0].Error,match='Supersession preimage differs'):
                cur.execute(broken)
            conn.rollback()
            cur.execute("SELECT to_regclass('public.lab_arena_scoring_supersessions'),EXISTS(SELECT 1 FROM pg_attribute WHERE attrelid='public.lab_arena_runs'::regclass AND attname='scoring_generation_floor' AND NOT attisdropped)")
            assert cur.fetchone()==(None,False)
            cur.execute("SELECT jsonb_build_array(oid,nspowner,nspacl::TEXT) FROM pg_namespace WHERE nspname='public'")
            assert cur.fetchone()[0]==before
            cur.execute(sql);cur.execute(sql)
            cur.execute("SELECT jsonb_build_array(oid,nspowner,nspacl::TEXT) FROM pg_namespace WHERE nspname='public'")
            assert cur.fetchone()[0]==before
            cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
            assert cur.fetchone()==(owner_already_has_create,)
            cur.execute("SELECT jsonb_agg(jsonb_build_array(defaclrole,defaclnamespace,defaclobjtype,defaclacl::TEXT) ORDER BY oid) FROM pg_default_acl")
            assert cur.fetchone()[0]==defaults
            cur.execute("SELECT has_table_privilege('leadpoet_parity_reader','public.lab_arena_scoring_supersessions','SELECT'),has_table_privilege('lab_arena_service','public.lab_arena_scoring_supersessions','SELECT')")
            assert cur.fetchone()==(False,True)
    finally:
        if admin is not None:
            with admin.cursor() as cur:cur.execute('ALTER ROLE postgres SUPERUSER')
            admin.close()
        gen.close()
