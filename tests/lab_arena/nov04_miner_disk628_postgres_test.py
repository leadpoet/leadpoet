"""A miner disk repair retains79 other positions and uses ordinary stage2 work."""
import base64
from copy import deepcopy
import json
import re
import threading

import pytest

from lab_arena import contracts, scoring, verify
from lab_arena.store import hash_lease_token
from scripts import prepare_nov4_miner_disk629 as preparation
from tests.lab_arena import nov04_baseline_source621_postgres_test as source
from tests.lab_arena.nov04_baseline_disk626_postgres_test import complete

ROUND=source.ROUND
SUB=preparation.SUBMISSION
OLD=f'{ROUND}:{SUB}:2:7:2'
NEW=preparation.SOURCE
scalar=source.scalar
SCRIPTS=source.SCRIPTS


@pytest.fixture
def database():
    generator=source.database.__wrapped__()
    try:yield next(generator)
    finally:generator.close()


prepared=source.prepared


def setup(p):
    source.apply(p)
    with p.conn.cursor() as cur:
        cur.execute((SCRIPTS/'625-lab-arena-verified-fit-checkpoint.sql').read_text())
        text=(SCRIPTS/'626-arena-2026-11-04-baseline-disk-recovery.sql').read_text()
        schema=text.split('DO $data626$')[0]
        restore='DO $restore626$'+text.split('DO $restore626$')[1].split('NOTIFY')[0]
        cur.execute(schema+restore+'COMMIT;')
    p.conn.commit();source.age_pending(p)
    source.finish_source_executions(p)
    p.service._recover_baseline(p.store.get_round(ROUND));source.age_pending(p)
    source.finish_source_judgments(p)
    assert p.service.close_scoring(ROUND,2)['status']=='closed'
    assert p.service.score_stage(ROUND,2)['status']=='ok'
    p.service._record_recovered_judgment_scores(p.store.get_round(ROUND))
    original='source621-miner-0'
    # Only fixture preparation uses replication bypass. Repair, execution,
    # billing, judging, plan updates, scoring and publication remain native.
    with p.conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_submissions SET submission_id=%s,code_review_status=\'passed\',code_review_attempts=1,code_review_doc=\'{}\',code_review_claim=\'sha256:\'||repeat(\'a\',64),code_review_started_at=now() WHERE submission_id=%s',(SUB,original))
        rows=scalar(cur,'SELECT jsonb_agg(to_jsonb(a)) FROM public.lab_arena_runs a WHERE submission_id=%s',(original,))
        for row in rows:
            changed=json.loads(json.dumps(row).replace(original,SUB))
            assignments=','.join(f'{key}=%s' for key in ('run_id','assignment_id','submission_id','scored_run_id','output_ref'))
            cur.execute('UPDATE public.lab_arena_runs SET '+assignments+' WHERE run_id=%s',tuple(changed[key]for key in ('run_id','assignment_id','submission_id','scored_run_id','output_ref'))+(row['run_id'],))
        cur.execute("UPDATE public.lab_arena_rounds SET participants=replace(participants::text,%s,%s)::jsonb,stage2_scoring_plan_doc=replace(stage2_scoring_plan_doc::text,%s,%s)::jsonb,finalists=replace(finalists::text,%s,%s)::jsonb WHERE round_id=%s",(original,SUB,original,SUB,original,SUB,ROUND))
        old=scalar(cur,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(OLD.removesuffix(':2')+':1',))
        replacement={**old,'run_id':OLD,'attempt':2,'output_ref':OLD+'.json'}
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='lease_expired',result_doc=NULL,output_ref=NULL,per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s",(old['run_id'],))
        cur.execute('INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*',(json.dumps(replacement),))
        cur.execute('UPDATE public.lab_arena_runs SET scored_run_id=%s WHERE assignment_id=%s',(OLD,old['assignment_id']+':score'))
        row=scalar(cur,"SELECT to_jsonb(a)-'updated_at' FROM public.lab_arena_rounds a WHERE round_id=%s",(ROUND,))
        plan=deepcopy(row['stage2_scoring_plan_doc'])
        for item in plan['work_items']:
            if item['submission_id']==SUB and item['icp_position']==7:item.update(scored_run_id=OLD,output_ref=OLD+'.json')
        cur.execute('UPDATE public.lab_arena_rounds SET stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s',(json.dumps(plan),ROUND))
        row=scalar(cur,"SELECT to_jsonb(a)-'updated_at' FROM public.lab_arena_rounds a WHERE round_id=%s",(ROUND,))
        # The626 data lifecycle has its own full native regression. This test
        # seeds its stable receipt after installing its exact schema prefix.
        cur.execute("INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc) SELECT %s,coalesce(max(generation_floor),0)+1,'nov04-baseline-disk626-v1',%s,'fixture626',txid_current(),pg_backend_pid(),%s::jsonb,%s::jsonb,'{}','{\"operation\":\"baseline_disk_recovery626_v1\",\"work_items\":[],\"superseded_runs\":{}}' FROM public.lab_arena_scoring_supersessions WHERE round_id=%s",(ROUND,'sha256:'+'d'*64,json.dumps(row),json.dumps(row),ROUND))
        for kind,content in [('runtime.stderr',{'text':'TYCHE runtime blocked: [Errno 28] No space left on device','sequence':0}),('runtime.finished',{'exit_code':0,'company_count':4})]:
            cur.execute("INSERT INTO public.lab_arena_trajectory_events(run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content) SELECT run_id,gen_random_uuid(),round_id,submission_id,miner_hotkey,miner_hotkey,assignment_id,icp_position::text,stage,icp_position,attempt,kind,'miner',%s,now(),%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s",(kind,json.dumps(content),OLD))
        cur.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    p.objects.update({k.replace(original,SUB):v.replace(original.encode(),SUB.encode())for k,v in list(p.objects.items())})
    from tests.lab_arena.company_judgments_test import _company
    p.objects[OLD+'.json']=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':[_company('Retained'+str(i))for i in range(4)]}).encode()
    judge=OLD.rsplit(':',1)[0]+':score:1'
    original_judge=json.loads(p.objects[judge+'.json']);original_judge['scored_run_id']=OLD;p.objects[judge+'.json']=json.dumps(original_judge).encode()


def render(p):
    with p.conn.cursor() as cur:
        manifest={'submission_hash':scalar(cur,"SELECT encode(extensions.digest((to_jsonb(s)-'updated_at')::TEXT,'sha256'),'hex') FROM public.lab_arena_submissions s WHERE submission_id=%s",(SUB,)),
            'miner_run_hashes':scalar(cur,"SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) FROM public.lab_arena_runs a WHERE round_id=%s AND stage=2",(ROUND,))}
    return re.sub(r'\$manifest628\$.*?\$manifest628\$',lambda _:'$manifest628$'+json.dumps(manifest,separators=(',',':'))+'$manifest628$',(SCRIPTS/'628-arena-2026-11-04-miner-disk-recovery.sql').read_text(),flags=re.S)


@pytest.mark.parametrize('outcome',['accepted','provider_error'])
def test_miner_repair_native_execute_judge_plan_and_preserved_work(prepared,outcome):
    p=prepared;setup(p)
    ordinary_before=p.store.icp_cost_eligibility(round_id=ROUND,submission_id=SUB,icp_position=6,qualified_company_count=0)
    with p.conn.cursor() as cur:
        # Existing settled liability is a fixture fact. The replacement's
        # provider lifecycle below uses native reserve/dispatch/settle RPCs.
        for kind,amount,doc,terminal in [
            ('reservation',100,{'action_sequence':1,'request_hash':'sha256:'+'e'*64},None),
            ('dispatch',100,{},None),
            ('settlement',6000000,{}, {'status':200,'headers':{},'body_b64':'e30=','call_succeeded':True}),
        ]:
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,round_id,run_id,submission_id,miner_hotkey,stage,provider,operation_id,funding_source,call_identity,amount_microusd,entry_doc,terminal_response) SELECT %s,round_id,run_id,submission_id,miner_hotkey,stage,'openrouter','openrouter.chat','miner_key',%s,%s,%s::jsonb,%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s",(kind,'sha256:'+'8'*64,amount,json.dumps(doc),json.dumps(terminal),OLD))

        before=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))
        financial=scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
    p.conn.commit()
    sql=render(p)
    bad_manifest=sql.replace(json.loads(re.search(r'\$manifest628\$(.*?)\$manifest628\$',sql,re.S)[1])['submission_hash'],'0'*64)
    with pytest.raises(p.pg.Error):
        with p.conn.cursor() as cur:cur.execute(bad_manifest)
    p.conn.rollback()
    with p.conn.cursor() as cur:
        assert scalar(cur,"SELECT to_regprocedure('public.lab_arena__miner_disk628_v1(public.lab_arena_runs)')") is None
        cur.execute(sql)
    p.conn.commit()
    with p.conn.cursor() as cur:
        after=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))
        assert len(after)==len(before)+1
        retired={OLD,OLD.rsplit(':',1)[0]+':score:1'}
        for key,row in before.items():
            omit={'status','terminal_cause','terminal_doc','updated_at'}if key in retired else set()
            assert {k:v for k,v in row.items()if k not in omit}=={k:v for k,v in after[key].items()if k not in omit}
        assert scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')==financial
        assert scalar(cur,'SELECT public.lab_arena__baseline_recovery_run_v1(a) FROM public.lab_arena_runs a WHERE run_id=%s',(NEW,)) is False
        cur.execute(sql)
        assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE run_id=%s',(NEW,))==1
    p.conn.commit()
    assert p.store.icp_cost_eligibility(round_id=ROUND,submission_id=SUB,icp_position=6,qualified_company_count=0)==ordinary_before
    p.service._lock=threading.RLock()
    with pytest.raises(ValueError):preparation.prepare(p.service)
    assert p.service._advance_round_locked(ROUND)['status']=='draining'
    assert p.service.publish(ROUND)['status']=='stale'
    source.age_pending(p)
    lease=source.claim(p);assert lease.get('run_id')==NEW,lease
    token=hash_lease_token('source621')
    def rpc(sql,args):
        with p.conn.cursor() as cur:
            cur.execute('SET ROLE lab_arena_service');cur.execute('SELECT public.'+sql,args)
            result=cur.fetchone()[0];cur.execute('RESET ROLE')
        p.conn.commit();return result
    funding=rpc('lab_arena_provider_funding(%s,%s)',(NEW,'openrouter'))['funding_source']
    identity='sha256:'+'9'*64
    reserve=lambda call:rpc("lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter',%s,100,%s::jsonb,1500)",(NEW,token,call,funding,json.dumps({'action_sequence':1,'request_hash':'sha256:'+'f'*64})))
    assert reserve(identity)['status']=='reserved'
    assert rpc('lab_arena_mark_dispatched(%s,%s,%s)',(NEW,token,identity))['status']=='dispatched'
    assert rpc('lab_arena_settle_call(%s,%s,%s,6000000,%s::jsonb,1500)',(NEW,token,identity,json.dumps({'status':200,'headers':{},'body_b64':'e30=','call_succeeded':True})))['status']=='settled'
    assert reserve('sha256:'+'a'*64)['reason']=='money_cap'
    with p.conn.cursor() as cur:
        costs=scalar(cur,'SELECT public.lab_arena__successful_icp_cost_state(%s,%s,7)',(ROUND,SUB))
        assert costs['settled_microusd']==12000000 and costs['competition_settled_microusd']==6000000
        assert costs['excluded_infrastructure_settled_microusd']==6000000
        assert scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l WHERE entry_id<=%s',(max(x['entry_id']for x in financial),))==financial

    if outcome=='provider_error':
        assert complete(p,lease,failed=True)['status']=='failed'
        p.store.expire_leases(ROUND)
        assert not any(r['attempt']>3 for r in p.store.list_runs(ROUND)if r['assignment_id']==lease['assignment_id'])
        assert source.claim(p)['status']=='no_pending'
        return
    assert complete(p,lease)['status']=='accepted'
    assert p.service._advance_round_locked(ROUND)['status']=='draining'
    assert p.service.publish(ROUND)['status']=='stale'
    # A changing backing object cannot change the canonical work item after
    # its bytes were hashed: preparation reads the target exactly once.
    backing=p.service._objects;reads=[]
    class ChangingOutput:
        def get(self,ref):return backing.get(ref)
        def get_bounded(self,ref,limit):
            if ref==lease['run_id']+'.json':
                reads.append(ref)
                if len(reads)>1:raise AssertionError('unbound second output read')
            return backing.get_bounded(ref,limit)
    p.service._objects=ChangingOutput()
    payload=preparation.prepare(p.service)
    p.service._objects=backing
    assert len(reads)==1

    plan_before=p.store.get_round(ROUND)['stage2_scoring_plan_doc']
    second=(SCRIPTS/'629-arena-2026-11-04-miner-replacement-judge.sql').read_text()
    wrong=deepcopy(payload);wrong['expected_round']['stage_generation']+=1
    with p.conn.cursor() as cur:cur.execute("SELECT set_config('lab_arena.miner_disk629_payload',%s,false)",(json.dumps(wrong),))
    with pytest.raises(p.pg.Error):
        with p.conn.cursor() as cur:cur.execute(second)
    p.conn.rollback()
    with p.conn.cursor() as cur:
        assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=2',(lease['assignment_id']+':score',))==0
        cur.execute("SELECT set_config('lab_arena.miner_disk629_payload',%s,false)",(json.dumps(payload),))
        cur.execute(second)
    p.conn.commit()
    with p.conn.cursor() as cur:
        cur.execute(second)
        assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND status=\'pending\'',(lease['assignment_id']+':score',))==1
    p.conn.commit()
    plan_after=p.store.get_round(ROUND)['stage2_scoring_plan_doc']
    assert sum(a!=b for a,b in zip(plan_before['work_items'],plan_after['work_items']))==1
    assert len(plan_after['work_items'])==80 and plan_before['zero_rows']==plan_after['zero_rows']
    assert p.service._advance_round_locked(ROUND)['status']=='waiting'
    source.age_pending(p);judge=source.claim(p)
    assert judge['kind']=='score' and judge['scored_run_id']==NEW and judge['attempt']==2
    assert complete(p,judge)['status']=='accepted'
    assert p.service.close_scoring(ROUND,2)['status']=='closed'
    assert p.service.score_stage(ROUND,2)['status']=='ok'
    from lab_arena.training_export import _competition_cost_snapshot
    run=p.store.get_run(NEW);row=p.store.get_round(ROUND)
    observed={'observed_at':p.service.now().isoformat(),'eligibility':p.store.icp_cost_eligibility(round_id=ROUND,submission_id=SUB,icp_position=7,qualified_company_count=0)}
    scored=verify.scored_row(SUB,7,NEW,p.service.evaluation_icps(ROUND)[7],[],[],row['configuration_doc']['scorer_policy'])
    cost=_competition_cost_snapshot(run=run,configuration=row['configuration_doc'],scored=scored,
        related=[r for r in p.store.list_runs(ROUND)if r['assignment_id']==run['assignment_id']or r.get('scored_run_id')==NEW],reader=lambda **_:observed)
    assert cost['status']=='available' and cost['competition_sourcing_microusd']==6000000,cost
    assert p.service.publish(ROUND)['status']=='ok'
    assert p.store.get_round(ROUND)['reward_activated_at'] is None
