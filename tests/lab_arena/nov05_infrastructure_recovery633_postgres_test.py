"""Approved recovery keeps571 historical rows and reuses165 judge identities.

All providers and outputs are synthetic. Only fixture construction uses replica
inserts; migration, claims, completions, plan rebuilding and publication use the
normal SQL/service paths on a disposable local PostgreSQL database.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

import pytest

from lab_arena import company_judgments, contracts, scoring
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import nov04_baseline_source621_postgres_test as source
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import prepare_current_service_reads
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL=source.SCRIPTS/'633-arena-2026-11-05-approved-infrastructure-recovery.sql'
ROUND='arena-2026-11-05'
TOKEN=hash_lease_token('recovery633-fixture')
scalar=source.scalar

@pytest.fixture(scope='module')
def database():
    g=source.database.__wrapped__();pg,dsn=next(g)
    try:
        with pg.connect(**dsn) as conn,conn.cursor() as cur:
            for name,tag,data in [('621-arena-2026-11-04-baseline-source-correction.sql','621','correct621'),('630-arena-2026-11-04-baseline-donor-evidence.sql','630','data630')]:
                text=(source.SCRIPTS/name).read_text()
                restore='DO $restore'+tag+'$'+text.split('DO $restore'+tag+'$')[1].split('NOTIFY')[0]
                cur.execute(text.split('DO $'+data+'$')[0]+restore+'COMMIT;')
        prepare_current_service_reads(lambda:pg.connect(**dsn))
        yield pg,dsn
    finally:g.close()


def seed(database):
    pg,dsn=database;conn=pg.connect(**dsn);objects={}
    # Use the existing production-compatible legacy company-judgment policy.
    _,context,_,_,_=source.prior.prior.context()
    cfg=source.base_round_configuration();cfg.update(context['configuration_doc'])
    cfg.update(round_id=ROUND,mode='live',network_name='finney',netuid=71,
        baseline_hotkey=hotkey('633-baseline'),stage_1_icp_count=5,stage_2_icp_count=5,
        execution_sequence_policy='baseline_scored_first_v1',infrastructure_retry_cost_policy='trusted_infrastructure_v1',
        max_attempts_per_assignment=2,execution_cap_microusd=60000000,
        cost_per_company_microusd=6000000,execution_icp_cap_microusd=6000000,
        sourcing_cost_eligibility_policy='successful_calls_flat_per_icp_v1')
    cfg.update(reasoning_capture_policy='full_generated_thinking_v1',reasoning_route_registry=source.approved_registry(),
        baseline_reasoning_configuration={'reasoning_capture_policy':'full_generated_thinking_v1','reasoning_route_registry':source.approved_registry()},
        baseline_source_url='https://github.com/leadpoet/leadpoet-sales-agent/archive/'+'3'*40+'.tar.gz',
        reasoning_baseline_source_sha256=contracts.hash_bytes(b'fixture633 admitted baseline source'))
    cfg['schedule']={k:v.replace('2026-09-01','2026-11-04').replace('2026-09-02','2026-11-05')for k,v in source.base_round_configuration()['schedule'].items()}
    cfg['schedule'].update(stage_2_close='2026-11-05T14:00:02Z',final_scoring_close='2026-11-05T20:30:02Z',publication_deadline='2026-11-05T20:30:03Z')
    cfg=contracts.validate_round_configuration(cfg)
    objects[f'arena/{ROUND}/sources/baseline-2026-11-05.tar.gz']=b'fixture633 admitted baseline source'
    participants=[dict(submission_id='baseline-2026-11-05',miner_hotkey=cfg['baseline_hotkey'],is_king=True)]
    participants += [dict(submission_id='miner633-'+str(i),miner_hotkey=hotkey('633-miner-'+str(i)),is_king=False)for i in range(24)]
    rows=[];targets=[];now=datetime.now(timezone.utc)
    def execution(p,pos,attempt=1,status='accepted',cause='accepted',stage=2,unused=False):
        aid=f'{ROUND}:{p["submission_id"]}:{stage}:{pos}';rid=f'{aid}:{attempt}'
        row=dict(run_id=rid,assignment_id=aid,round_id=ROUND,submission_id=p['submission_id'],miner_hotkey=p['miner_hotkey'],stage=stage,icp_position=pos,attempt=attempt,kind='execute',status=status,stage_generation=4,scoring_generation_floor=0,terminal_cause=cause,lease_generation=0)
        if status=='accepted':
            row.update(output_ref=rid+'.json',result_doc={'terminal_status':'accepted'})
            objects[row['output_ref']]=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':[]}).encode()
        elif unused:
            row.update(terminal_doc={'closed_at':'2026-11-05T14:23:51Z','previous_status':'pending','infrastructure_incomplete':True},lease_generation=1 if attempt==2 else 0)
        elif cause=='lease_expired':
            row.update(runner_hotkey=hotkey('633-old-runner'),lease_generation=attempt,lease_token_hash=TOKEN,lease_expires_at=(now-timedelta(hours=3)).isoformat(),claim_request_id=uuid.uuid4().hex,claim_request_hash='sha256:'+'a'*64,claim_response={'status':'leased','lease_generation':attempt,'lease_expires_at':(now-timedelta(hours=3)).isoformat()})
        rows.append(row);return row
    def judge(e,attempt,status,cause):
        aid=e['assignment_id']+':score';rid=aid+':'+str(attempt)
        row=dict(run_id=rid,assignment_id=aid,round_id=ROUND,submission_id=e['submission_id'],miner_hotkey=e['miner_hotkey'],stage=e['stage'],icp_position=e['icp_position'],attempt=attempt,kind='score',status=status,stage_generation=7,scoring_generation_floor=0,terminal_cause=cause,scored_run_id=e['run_id'],company_judgment_refs=[],result_doc={'terminal_status':cause,'company_judgment_consumed_indexes':[]},claim_response={'company_judgment_cache':{'schema_version':company_judgments.LEASE_SCHEMA_VERSION,'hits':[],'misses':[]}})
        if status=='accepted':
            row['output_ref']=rid+'.json';objects[row['output_ref']]=json.dumps({'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':e['run_id'],'breakdowns':[],'company_judgments':[]}).encode()
        rows.append(row)
    for pos in range(10):judge(execution(participants[0],pos,stage=1),1,'accepted','accepted')
    for slot in range(240):
        p=participants[1+slot//10];pos=slot%10
        if slot<165:
            if slot<45:execution(p,pos,status='failed',cause='provider_error'if slot<2 else'lease_expired')
            e=execution(p,pos,attempt=2 if slot<45 else 1)
            if slot<148:
                if slot<14:judge(e,1,'failed','lease_expired'if slot<9 else'judge_error')
                judge(e,2 if slot<14 else 1,'accepted','accepted')
            else:
                for attempt in range(1,6):judge(e,attempt,'failed','judge_error'if slot==164 or slot==148 and attempt<=2 else'credential_error')
        else:
            missing=slot-165
            if missing<36:
                if missing<13:execution(p,pos,status='failed',cause='lease_expired')
                e=execution(p,pos,attempt=2 if missing<13 else 1,status='failed',cause='stage_closed',unused=True);mode='restore_unused'
            elif missing<50:e=execution(p,pos,status='failed',cause='lease_expired');mode='ordinary_retry'
            elif missing<55:
                execution(p,pos,status='failed',cause='lease_expired');execution(p,pos,attempt=2,status='failed',cause='lease_expired');continue
            elif missing<74:
                e=execution(p,pos,status='failed',cause='credential_error');e['result_doc']={'terminal_status':'credential_error'};continue
            else:
                execution(p,pos,status='failed',cause='provider_error');execution(p,pos,attempt=2,status='failed',cause='provider_error');continue
            targets.append({'run_id':e['run_id'],'mode':mode})
    assert len(rows)==571 and sum(x['status']=='accepted'for x in rows)==333
    plans={}
    for stage in (1,2):plans[stage]=scoring.build_scoring_plan(round_id=ROUND,stage=stage,runs=[x for x in rows if x['kind']=='execute'and x['stage']==stage],execution_sequence_policy='baseline_scored_first_v1',configuration=cfg)
    with conn.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]' WHERE singleton")
        q.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc,participants,evaluation_date,icp_set_date,benchmark_ref,stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists) VALUES (%s,'stage2_judged',8,7,%s::jsonb,%s::jsonb,'2026-11-05','2026-11-04','benchmark633.json',%s::jsonb,%s::jsonb,%s::jsonb)",(ROUND,json.dumps(cfg),json.dumps(participants),json.dumps(plans[1]),json.dumps(plans[2]),json.dumps([x['submission_id']for x in participants[1:]])))
        future=deepcopy(cfg);future['round_id']='arena-2026-11-06';q.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,evaluation_date) VALUES ('arena-2026-11-06','open',%s::jsonb,'2026-11-06')",(json.dumps(future),))
        for p in participants:
            q.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,is_king,status,source_ref,source_size_bytes,source_archive_sha256,source_normalized_sha256,code_review_status,code_review_attempts,code_review_doc,code_review_claim,code_review_started_at) VALUES (%s,%s,%s,%s,'frozen',%s,100,%s,%s,'passed',1,'{}',%s,now())",(p['submission_id'],ROUND,p['miner_hotkey'],p['is_king'],f'arena/{ROUND}/sources/{p["submission_id"]}.tar.gz','sha256:'+'4'*64,'sha256:'+'5'*64,'sha256:'+'6'*64))
        for row in rows:
            cols=list(row);q.execute('INSERT INTO public.lab_arena_runs('+','.join(cols)+') VALUES('+','.join('%s::jsonb'if isinstance(row[k],(dict,list))else'%s'for k in cols)+')',[json.dumps(row[k])if isinstance(row[k],(dict,list))else row[k]for k in cols])
        for e in rows:
            if e['kind']=='execute'and e['status']=='accepted':
                identity='sha256:'+hashlib.sha256(e['run_id'].encode()).hexdigest()
                for kind,doc,terminal in [('reservation',{'action_sequence':1,'request_hash':'sha256:'+'e'*64},None),('dispatch',{},None),('settlement',{}, {'status':200,'headers':{},'body_b64':'e30=','call_succeeded':True})]:
                    q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,round_id,run_id,submission_id,miner_hotkey,stage,provider,operation_id,funding_source,call_identity,amount_microusd,entry_doc,terminal_response) VALUES (%s,%s,%s,%s,%s,%s,'openrouter','openrouter.chat','miner_key',%s,1,%s::jsonb,%s::jsonb)",(kind,ROUND,e['run_id'],e['submission_id'],e['miner_hotkey'],e['stage'],identity,json.dumps(doc),json.dumps(terminal)if terminal else None))
        q.execute('SET LOCAL session_replication_role=origin')
    conn.commit();store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
    svc=ArenaService.__new__(ArenaService);svc._store=store;svc._objects=type('Objects',(),{'get':lambda _,k:objects[k],'get_bounded':lambda _,k,n:objects[k]})();svc._round=lambda rid:store.get_round(rid);svc._clock=lambda:datetime(2026,11,5,18,tzinfo=timezone.utc);svc._config=SimpleNamespace(mode='live',publication_lock_dir=None);svc.evaluation_icps=lambda _rid:daily_icps()[:10];svc.benchmark_icps=lambda _rid:daily_icps()[:10]
    p=SimpleNamespace(pg=pg,dsn=dsn,conn=conn,store=store,service=svc,objects=objects,targets=targets,participants=participants)
    p.sql=render(p);return p


def render(p):
    with p.conn.cursor()as q:
        row=scalar(q,'SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(ROUND,))
        m=dict(schema_version='leadpoet.lab_arena.nov05_recovery633.v1',round_id=ROUND,status_generation=row['status_generation'],stage_generation=row['stage_generation'],configuration_sha256=source.hash_doc(q,row['configuration_doc']),plan_sha256=source.hash_doc(q,row['stage2_scoring_plan_doc']),sources={},targets=[])
        for t in p.targets:
            old=scalar(q,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(t['run_id'],));m['targets'].append({**t,'row_sha256':source.hash_doc(q,old)})
            s=scalar(q,"SELECT jsonb_build_object('round_id',round_id,'submission_id',submission_id,'miner_hotkey',miner_hotkey,'source_ref',source_ref,'source_size_bytes',source_size_bytes,'source_archive_sha256',source_archive_sha256) FROM public.lab_arena_submissions WHERE submission_id=%s",(old['submission_id'],));m['sources'][old['submission_id']]=source.hash_doc(q,s)
    text=re.sub(r'\$manifest633\$.*?\$manifest633\$',lambda _:'$manifest633$'+json.dumps(m,separators=(',',':'))+'$manifest633$',SQL.read_text(),flags=re.S)
    # Only disposable test dates move. Production's Nov5 scope stays literal.
    return text.replace('2026-11-05T20:30:00Z','2099-11-05T20:30:00Z').replace('2026-11-05T23:58:59Z','2099-11-05T23:58:59Z').replace('2026-11-05T23:59:00Z','2099-11-05T23:59:00Z')


def apply(p,sql=None):
    with p.conn.cursor()as q:q.execute(sql or p.sql)
    p.conn.commit()


def snapshot(p):
    with p.conn.cursor()as q:return dict(runs=scalar(q,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a'),rounds=scalar(q,'SELECT jsonb_object_agg(round_id,to_jsonb(a)) FROM public.lab_arena_rounds a'),ledger=scalar(q,'SELECT coalesce(jsonb_agg(to_jsonb(l) ORDER BY entry_id),\'[]\'::jsonb) FROM public.lab_arena_ledger l'),triggers=scalar(q,"SELECT jsonb_agg(jsonb_build_array(tgname,tgenabled,pg_get_triggerdef(oid)) ORDER BY tgname) FROM pg_trigger WHERE tgrelid='public.lab_arena_runs'::regclass AND NOT tgisinternal"))


def claim(p):
    with p.conn.cursor()as q:
        q.execute('SET ROLE lab_arena_service');q.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,1500)',(ROUND,hotkey('633-runner'),uuid.uuid4().hex,'sha256:'+'b'*64,TOKEN));v=q.fetchone()[0];q.execute('RESET ROLE')
    p.conn.commit();return v


def complete(p,lease,*,cause='accepted'):
    score=lease['kind']=='score';out={'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':lease['scored_run_id'],'breakdowns':[],'company_judgments':[]}if score else{'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':[]};raw=json.dumps(out).encode();ref=lease['run_id']+'.json';p.objects[ref]=raw
    return p.store.complete_attempt(run_id=lease['run_id'],lease_token_hash=TOKEN,result={'terminal_status':cause,**({'company_judgment_consumed_indexes':[]}if score else{})},terminal_cause=cause,output_ref=ref if cause=='accepted'else None,output_hash=contracts.hash_bytes(raw)if cause=='accepted'else None,**({'company_judgment_evidence':[]}if score and cause=='accepted'else{}),completion_request_hash=contracts.document_hash({'run':lease['run_id']}))


@pytest.fixture
def prepared(database):
    p=seed(database)
    try:yield p
    finally:p.store.close();p.conn.close()


def test_exact_571_row_recovery_replay_and_native_merged_scoring(prepared):
    p=prepared;before=snapshot(p);assert len(before['runs'])==571
    apply(p);row=p.store.get_round(ROUND)
    assert row['status']=='stage2'and row['stage2_scoring_plan_doc']is None and row['scoring_generation_floor']==0
    after=snapshot(p);assert sum(x['status']=='pending'for x in after['runs'].values())==50
    assert len(after['runs'])==585
    assert after['triggers']==before['triggers']and after['ledger']==before['ledger']
    assert after['rounds']['arena-2026-11-06']==before['rounds']['arena-2026-11-06']
    assert {k:v for k,v in after['runs'].items()if v['status']=='accepted'}=={k:v for k,v in before['runs'].items()if v['status']=='accepted'}
    apply(p);assert snapshot(p)==after
    # A stale driver request cannot open the old165-item plan in execution.
    old=before['rounds'][ROUND]['stage2_scoring_plan_doc']
    assert p.store.open_scoring(ROUND,2,[{**x,'company_judgment_refs':[]}for x in old['work_items']],company_quality_cache=True)['status']=='existing'
    assert snapshot(p)==after
    restored_second=False
    for _ in range(50):
        lease=claim(p);assert lease['status']=='leased'and lease['kind']=='execute',lease
        restored_second |= lease['attempt']==2 and lease['run_id']in{x['run_id']for x in p.targets if x['mode']=='restore_unused'}
        bill(p,lease)
        assert complete(p,lease)['status']=='accepted'
    assert restored_second
    assert p.service.close_stage(ROUND,2)['status']=='ok'
    merged=p.store.get_round(ROUND)['stage2_scoring_plan_doc'];assert len(merged['work_items'])==215
    assert all(x in merged['work_items']for x in old['work_items'])
    old_scores={k:v for k,v in before['runs'].items()if v['kind']=='score'}
    items=p.service._scoring_work_items(p.store.get_round(ROUND),2,merged)
    with p.conn.cursor()as q:
        q.execute('SAVEPOINT old_open')
        current=scalar(q,"SELECT pg_get_functiondef('public.lab_arena_open_scoring_v3(text,smallint,jsonb)'::regprocedure)")
        text=SQL.read_text();new=re.findall(r'\$new633\$(.*?)\$new633\$',text,re.S)[0];old_fragment=re.findall(r'\$old633\$(.*?)\$old633\$',text,re.S)[0]
        assert new in current
        q.execute(current.replace(new,old_fragment))
        with pytest.raises(p.pg.errors.UniqueViolation):q.execute('SELECT public.lab_arena_open_scoring_v3(%s,2::smallint,%s::jsonb)',(ROUND,json.dumps(items)))
        q.execute('ROLLBACK TO SAVEPOINT old_open')
        for wrong in (items+[items[0]],[{**items[0],'output_ref':'changed.json'}]):
            q.execute('SAVEPOINT invalid_open')
            with pytest.raises(p.pg.Error):q.execute('SELECT public.lab_arena_open_scoring_v3(%s,2::smallint,%s::jsonb)',(ROUND,json.dumps(wrong)))
            q.execute('ROLLBACK TO SAVEPOINT invalid_open')
    p.conn.commit()
    barrier=Barrier(2)
    def concurrent_open():
        with p.pg.connect(**p.dsn)as conn,conn.cursor()as q:
            q.execute('SET ROLE lab_arena_service');barrier.wait(timeout=5)
            return scalar(q,'SELECT public.lab_arena_open_scoring_v3(%s,2::smallint,%s::jsonb)',(ROUND,json.dumps(items)))
    with ThreadPoolExecutor(max_workers=2)as pool:opened=list(pool.map(lambda _:concurrent_open(),range(2)))
    assert sorted(x['status']for x in opened)==['existing','ok']
    assert next(x['assignments']for x in opened if x['status']=='ok')==50
    assert {k:p.store.get_run(k)for k in old_scores}==old_scores
    assert p.service.scoring_is_complete(ROUND,2)is False
    for i in range(50):
        lease=claim(p);assert lease['status']=='leased'and lease['kind']=='score'and lease['attempt']==1,lease
        assert complete(p,lease)['status']=='accepted'
        if i==48:assert p.service.scoring_is_complete(ROUND,2)is False
    assert p.service.scoring_is_complete(ROUND,2)is True
    progressed=snapshot(p);apply(p);assert snapshot(p)==progressed
    assert p.service.close_scoring(ROUND,2)['status']=='closed'
    # The inherited17 exhausted assignments stay unmeasured, never receive6.
    assert not any(x['attempt']>5 for x in p.store.list_runs(ROUND,kind='score'))
    assert p.service.score_stage(ROUND,2)['remaining']==17
    # Move only the disposable clock boundary after native work has finished.
    # Production keeps the approved literal schedule; no clock wait is needed.
    now=datetime.now(timezone.utc)
    with p.conn.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        cfg=p.store.get_round(ROUND)['configuration_doc']
        cfg['schedule'].update(stage_2_close=(now-timedelta(seconds=3)).isoformat(),final_scoring_close=(now-timedelta(seconds=2)).isoformat(),publication_deadline=(now-timedelta(seconds=1)).isoformat())
        q.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',(json.dumps(cfg),ROUND))
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit();p.service._clock=lambda:now
    assert p.service.score_stage(ROUND,2)['status']=='ok'
    assert p.service.publish(ROUND)['status']=='ok'
    final=p.store.get_round(ROUND)
    assert final['status']=='published'and final['scoring_generation_floor']==0
    assert final['publication_doc']['king_decision']['outcome']=='no_king'
    assert {k:p.store.get_run(k)for k in old_scores}==old_scores
    assert final['publication_doc']['final_ranking']



def bill(p,lease):
    funding=p.store.provider_funding(lease['run_id'],'openrouter')['funding_source']
    identity='sha256:'+hashlib.sha256(lease['run_id'].encode()).hexdigest()
    assert p.store.reserve_call(run_id=lease['run_id'],lease_token_hash=TOKEN,call_identity=identity,operation_id='openrouter.chat',provider='openrouter',funding_source=funding,amount_microusd=1,call_doc={'action_sequence':1,'request_hash':'sha256:'+'e'*64},lease_ttl_seconds=1500)['status']=='reserved'
    assert p.store.mark_dispatched(run_id=lease['run_id'],lease_token_hash=TOKEN,call_identity=identity)['status']=='dispatched'
    assert p.store.settle_call(run_id=lease['run_id'],lease_token_hash=TOKEN,call_identity=identity,actual_microusd=1,terminal_response={'status':200,'headers':{},'body_b64':'e30=','call_succeeded':True},lease_ttl_seconds=1500)['status']=='settled'


@pytest.mark.parametrize('fault',['used_ledger','used_runtime','active','exhausted','successful_cap','unresolved_success','source_changed','terminal_guard_disabled'])
def test_recovery_rejects_unsafe_state_without_partial_changes(prepared,fault):
    p=prepared;unused=next(x['run_id']for x in p.targets if x['mode']=='restore_unused');expired=next(x['run_id']for x in p.targets if x['mode']=='ordinary_retry')
    with p.conn.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        if fault=='used_ledger':q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,round_id,run_id,miner_hotkey,amount_microusd,entry_doc) SELECT 'settlement',round_id,run_id,miner_hotkey,1,'{}' FROM public.lab_arena_runs WHERE run_id=%s",(unused,))
        elif fault=='used_runtime':q.execute("INSERT INTO public.lab_arena_trajectory_events(run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content) SELECT run_id,gen_random_uuid(),round_id,submission_id,miner_hotkey,miner_hotkey,assignment_id,icp_position::text,stage,icp_position,attempt,kind,'miner','runtime.started',now(),'{}' FROM public.lab_arena_runs WHERE run_id=%s",(unused,))
        elif fault=='active':q.execute("UPDATE public.lab_arena_runs SET status='leased' WHERE run_id=%s",(expired,))
        elif fault=='exhausted':
            bad=scalar(q,"SELECT run_id FROM public.lab_arena_runs WHERE kind='execute' AND attempt=2 AND terminal_cause='lease_expired' AND run_id<>ALL(%s) LIMIT 1",([x['run_id']for x in p.targets],))
            next(x for x in p.targets if x['mode']=='ordinary_retry')['run_id']=bad
        elif fault in('successful_cap','unresolved_success'):
            identity='sha256:'+'f'*64
            for kind,amount,doc,terminal in [('reservation',6000000,{'action_sequence':1,'request_hash':'sha256:'+'e'*64},None),('dispatch',6000000,{},None),('settlement'if fault=='successful_cap'else'uncertain',6000000,{'call':{'call_succeeded':True}}, {'status':200,'headers':{},'body_b64':'e30=','call_succeeded':True}if fault=='successful_cap'else None)]:
                q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,round_id,run_id,submission_id,miner_hotkey,stage,provider,operation_id,funding_source,call_identity,amount_microusd,entry_doc,terminal_response) SELECT %s,round_id,run_id,submission_id,miner_hotkey,stage,'openrouter','openrouter.chat','miner_key',%s,%s,%s::jsonb,%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s",(kind,identity,amount,json.dumps(doc),json.dumps(terminal)if terminal else None,expired))
        elif fault=='source_changed':q.execute("UPDATE public.lab_arena_submissions SET source_ref=replace(source_ref,'.tar.gz','-changed.tar.gz') WHERE submission_id=(SELECT submission_id FROM public.lab_arena_runs WHERE run_id=%s)",(unused,))
        elif fault=='terminal_guard_disabled':q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    if fault=='exhausted':p.sql=render(p)
    before=snapshot(p)
    try:
        with pytest.raises(p.pg.Error):apply(p)
        p.conn.rollback();assert snapshot(p)==before
        with p.conn.cursor()as q:assert scalar(q,"SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE round_id=%s AND request_id='nov05-approved-infrastructure633-v1'",(ROUND,))==0
    finally:
        if fault=='terminal_guard_disabled':
            p.conn.rollback()
            with p.conn.cursor()as q:q.execute('ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal')
            p.conn.commit()


def test_failure_after_terminal_suspension_rolls_back_every_change(prepared):
    p=prepared;before=snapshot(p)
    with p.conn.cursor()as q:definitions=scalar(q,"SELECT jsonb_object_agg(proname,pg_get_functiondef(oid)) FROM pg_proc WHERE oid=ANY(ARRAY['public.lab_arena_open_scoring_v3(text,smallint,jsonb)'::regprocedure,'public.lab_arena_scoring_policy_history_v1(text)'::regprocedure,'public.lab_arena_runs_terminal_v1()'::regprocedure])")
    broken=p.sql.replace('GET DIAGNOSTICS restored=ROW_COUNT;',"GET DIAGNOSTICS restored=ROW_COUNT; RAISE EXCEPTION 'fixture-after-disable';")
    with pytest.raises(p.pg.Error,match='fixture-after-disable'):apply(p,broken)
    p.conn.rollback();assert snapshot(p)==before
    with p.conn.cursor()as q:
        assert scalar(q,"SELECT jsonb_object_agg(proname,pg_get_functiondef(oid)) FROM pg_proc WHERE oid=ANY(ARRAY['public.lab_arena_open_scoring_v3(text,smallint,jsonb)'::regprocedure,'public.lab_arena_scoring_policy_history_v1(text)'::regprocedure,'public.lab_arena_runs_terminal_v1()'::regprocedure])")==definitions
        assert scalar(q,"SELECT tgenabled FROM pg_trigger WHERE tgrelid='public.lab_arena_runs'::regclass AND tgname='lab_arena_runs_terminal'")=='O'


def test_restored_second_attempt_failure_cannot_create_third(prepared):
    p=prepared;apply(p)
    targets={x['run_id']for x in p.targets if x['mode']=='restore_unused'and x['run_id'].endswith(':2')}
    with p.conn.cursor()as q:
        assert scalar(q,'SELECT count(*) FROM public.lab_arena_runs a WHERE run_id=ANY(%s) AND public.lab_arena_execution_attempt_count(a)=2',(list(targets),))==13
    p.conn.commit()
    for _ in range(50):
        lease=claim(p);assert lease['status']=='leased'
        if lease['run_id']in targets:break
        assert complete(p,lease)['status']=='accepted'
    else:pytest.fail('no restored physical second attempt was claimable')
    assert complete(p,lease,cause='provider_error')['status']=='failed'
    p.store.expire_leases(ROUND)
    assert sorted(x['attempt']for x in p.store.list_runs(ROUND,submission_id=lease['submission_id'],kind='execute')if x['assignment_id']==lease['assignment_id'])==[1,2]


def test_historical_round_preservation_exceeds_jsonb_aggregate_limit(prepared):
    """Hash each full row before aggregation; never combine historical payloads."""
    p=prepared
    old="SELECT jsonb_object_agg(a.round_id,to_jsonb(a)) FROM public.lab_arena_rounds a WHERE a.round_id<>%s"
    hashes="SELECT jsonb_object_agg(a.round_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) FROM public.lab_arena_rounds a WHERE a.round_id<>%s"
    with p.conn.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        # Seventeen independently valid 16MiB documents exceed the 256MiB
        # aggregate limit. PostgreSQL builds them; no large Python payloads.
        q.execute("""INSERT INTO public.lab_arena_rounds
            (round_id,status,configuration_doc,evaluation_date,stage1_scoring_plan_doc)
            SELECT 'arena-2026-10-'||lpad(i::text,2,'0'),'open',
                jsonb_set(configuration_doc,'{round_id}',to_jsonb('arena-2026-10-'||lpad(i::text,2,'0'))),
                '2026-10-01',jsonb_build_object('retained',repeat(chr(97+i),16*1024*1024))
            FROM public.lab_arena_rounds CROSS JOIN generate_series(1,17)i
            WHERE round_id=%s""",(ROUND,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    with p.conn.cursor()as q:
        assert scalar(q,"SELECT sum(octet_length(to_jsonb(a)::TEXT)) FROM public.lab_arena_rounds a WHERE round_id LIKE 'arena-2026-10-%%'")>256*1024*1024
        q.execute('SAVEPOINT old_aggregate')
        with pytest.raises(p.pg.errors.ProgramLimitExceeded,match='total size of jsonb object elements'):
            q.execute(old,(ROUND,))
        q.execute('ROLLBACK TO SAVEPOINT old_aggregate')
        before=scalar(q,hashes,(ROUND,))
        assert len(before)==18 and all(len(value)==64 for value in before.values())
        # Full-row hashes bind payload, other columns, and every row identity.
        q.execute('SET LOCAL session_replication_role=replica')
        for mutation in [
            "UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-02' WHERE round_id='arena-2026-10-01'",
            "UPDATE public.lab_arena_rounds SET stage1_scoring_plan_doc=jsonb_set(stage1_scoring_plan_doc,'{retained}','\"changed\"') WHERE round_id='arena-2026-10-01'",
            "UPDATE public.lab_arena_rounds SET round_id='arena-2026-09-01' WHERE round_id='arena-2026-10-01'",
            "DELETE FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01'",
        ]:
            q.execute('SAVEPOINT changed_history')
            q.execute(mutation)
            assert scalar(q,hashes,(ROUND,))!=before
            q.execute('ROLLBACK TO SAVEPOINT changed_history')
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    # The actual migration must apply and replay with all historical bytes
    # represented by exactly the same full-row digest map.
    apply(p)
    with p.conn.cursor()as q:
        assert scalar(q,hashes,(ROUND,))==before
        assert scalar(q,"SELECT count(*) FROM public.lab_arena_runs WHERE status='accepted'")==333
        assert scalar(q,"SELECT count(*) FROM public.lab_arena_runs WHERE kind='execute' AND status='pending'")==50
    p.conn.commit()
    apply(p)
    with p.conn.cursor()as q:assert scalar(q,hashes,(ROUND,))==before
