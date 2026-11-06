"""Exact sixth uses four retained judgments and cannot append a seventh."""
from datetime import datetime,timedelta,timezone
import json
import re
import hashlib
import uuid
import pytest

from lab_arena import company_judgments, contracts, scoring
from tests.lab_arena import nov05_infrastructure_recovery633_postgres_test as recovery
from tests.lab_arena import nov05_recovery_window641_postgres_test as window
from tests.lab_arena.company_judgments_test import _input, _raw
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SCRIPTS=recovery.source.SCRIPTS
SQL=SCRIPTS/'639-arena-2026-11-05-b66-sixth-judge.sql'
ROUND=recovery.ROUND
SUB='sub-b66ad20113a4402633ba875964ed4b80'
ASSIGNMENT=f'{ROUND}:{SUB}:2:3:score'
SOURCE=f'{ROUND}:{SUB}:2:3:1'
RUN=ASSIGNMENT+':6'
GRADING='sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b'
RUNTIME='sha256:0d930b667246e37886a5237ce98ed7838c2a8fefd430d124f2905aa4d19f18db'
REPOSITORY='493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model'
CLAIM='public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
HISTORY='public.lab_arena_scoring_policy_history_v1(text)'
HELPER='public.lab_arena__nov05_b66_judge639_v1(public.lab_arena_rounds,public.lab_arena_runs)'
scalar=recovery.scalar


def claim(p):
    runner=hotkey('633-runner')
    with p.conn.cursor()as q:
        q.execute('SET ROLE lab_arena_service')
        q.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,%s::text[],%s,%s,%s,1500)',(ROUND,runner,[runner],uuid.uuid4().hex,'sha256:'+'b'*64,recovery.TOKEN))
        result=q.fetchone()[0];q.execute('RESET ROLE')
    p.conn.commit();return result


@pytest.fixture(scope='module')
def database():
    gen=recovery.database.__wrapped__();pg,dsn=next(gen)
    try:
        with pg.connect(**dsn)as c,c.cursor()as q:
            # Install only committed historical helper and claim fragments.
            # Dated data corrections never run in this fixture.
            for n in (610,612,613,614):
                text=next(SCRIPTS.glob(str(n)+'-*.sql')).read_text()
                restore='DO $restore'+str(n)+'$'+text.split('DO $restore'+str(n)+'$')[1].split('NOTIFY')[0]
                q.execute(text.split('DO $patch'+str(n)+'$')[0]+restore+'COMMIT;');c.commit()
                definition=scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(CLAIM,))
                pairs=re.findall(r'\$old'+str(n)+r'\$(.*?)\$old'+str(n)+r'\$,\$new'+str(n)+r'\$(.*?)\$new'+str(n)+r'\$',text,re.S)
                old,new=next((a,b)for a,b in pairs if 'v_round,runs' in a)
                assert definition.count(old)==1;q.execute(definition.replace(old,new));c.commit()
            text=(SCRIPTS/'627-lab-arena-champion-baseline-freshness.sql').read_text()
            end=text.index('CREATE TEMP TABLE',text.index('$current627$;'))
            restore='DO $restore627$'+text.split('DO $restore627$')[1].split('NOTIFY')[0]
            q.execute(text[:end]+restore+'COMMIT;');c.commit()
            definition=scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(CLAIM,))
            for old,new in re.findall(r"\('public.lab_arena_claim_assignment[^']*','[^']*','[^']*',\d+,\$old627\$(.*?)\$old627\$,\$new627\$(.*?)\$new627\$\)",text,re.S):
                assert definition.count(old)==1;definition=definition.replace(old,new)
            q.execute(definition);c.commit()
            text=next(SCRIPTS.glob('635-*.sql')).read_text()
            q.execute('DO $patch635_history$'+text.split('DO $patch635_history$')[1].split('$patch635_history$;')[0]+'$patch635_history$;');c.commit()
            q.execute(recovery.SQL.read_text().split('-- Exact one-time data correction.')[0]+'COMMIT;');c.commit()
            for name,expected in [('lab_arena_complete_attempt','6c5ee308109b55b3b23ee1f1631d894cd72b8582ad1180c898e25f7ae9106ac0'),('lab_arena_expire_leases','b5887c1e2afa6ac5ff342f1dc4ed1f0fc390b92b7302533a3ba31db964dbfaca'),('lab_arena__score_attempt_count_v1','41f2a3f5fa3a5abf0e6106df52e98c59121a55e99e22daad88673ff8f45d9b84')]:
                assert scalar(q,"SELECT encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex') FROM pg_proc p WHERE proname=%s",(name,))==expected
            definitions={sig:scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(sig,))for sig in (CLAIM,HISTORY)}
            assert hashlib.sha256(definitions[CLAIM].encode()).hexdigest()=='a5dc7e6f4370dbffa6e390609a584e5d317cdba33445e58b5bf58bd8ec9f1f6c'
            assert hashlib.sha256(definitions[HISTORY].encode()).hexdigest()=='e56c877e3c2ba5440b2fc7b55caa65b2e94935c5117dddd12ed1320c9e3f4a45'
        yield pg,dsn,definitions
    finally:gen.close()


def snapshot(p):
    result=recovery.snapshot(p)
    with p.conn.cursor()as q:
        for table in ('lab_arena_company_judgments','lab_arena_signal_judgments'):
            result[table]=scalar(q,'SELECT coalesce(jsonb_agg(to_jsonb(a) ORDER BY cache_key),\'[]\') FROM public.'+table+' a')
        result['schema_acl']=scalar(q,"SELECT nspacl FROM pg_namespace WHERE nspname='public'")
        result['metadata']=scalar(q,"SELECT jsonb_object_agg(proname,to_jsonb(p)-'prosrc') FROM pg_proc p WHERE proname IN('lab_arena_claim_assignment','lab_arena_scoring_policy_history_v1','lab_arena__score_attempt_count_v1','lab_arena_complete_attempt','lab_arena_expire_leases','lab_arena_runs_terminal_v1','lab_arena_rounds_write_once_v1')")
        result['unchanged_functions']=scalar(q,"SELECT jsonb_object_agg(proname,pg_get_functiondef(p.oid)) FROM pg_proc p WHERE proname IN('lab_arena__score_attempt_count_v1','lab_arena_complete_attempt','lab_arena_expire_leases','lab_arena_runs_terminal_v1','lab_arena_rounds_write_once_v1')")
    return result


@pytest.fixture
def prepared(database):
    pg,dsn,definitions=database
    with pg.connect(**dsn)as c,c.cursor()as q:
        for d in definitions.values():q.execute(d)
        q.execute('TRUNCATE public.lab_arena_company_judgments,public.lab_arena_signal_judgments CASCADE')
    p=recovery.seed((pg,dsn));old='miner633-16'
    try:
        with p.conn.cursor()as q:
            q.execute('SET LOCAL session_replication_role=replica')
            for table,cols in [('lab_arena_runs',['run_id','assignment_id','submission_id','scored_run_id','output_ref']),('lab_arena_submissions',['submission_id','source_ref']),('lab_arena_ledger',['run_id','submission_id'])]:
                for col in cols:q.execute('UPDATE public.'+table+' SET '+col+'=replace('+col+',%s,%s)',(old,SUB))
            for col in ('participants','stage1_scoring_plan_doc','stage2_scoring_plan_doc','finalists'):
                q.execute('UPDATE public.lab_arena_rounds SET '+col+'=replace('+col+'::text,%s,%s)::jsonb WHERE round_id=%s',(old,SUB,ROUND))
            cfg=scalar(q,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))
            cfg.update(scorer_image_digest=GRADING,scorer_image_reference=REPOSITORY+'@'+GRADING)
            q.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',(json.dumps(cfg),ROUND))
            q.execute('SET LOCAL session_replication_role=origin')
        p.conn.commit()
        for t in p.targets:t['run_id']=t['run_id'].replace(old,SUB)
        objects={k.replace(old,SUB):v.replace(old.encode(),SUB.encode())for k,v in p.objects.items()};p.objects.clear();p.objects.update(objects)
        source=p.store.get_run(SOURCE);p.objects[source['output_ref']]=contracts.canonical_json({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':_input()['companies']}).encode()
        p.service.evaluation_icps=lambda _:[_input()['icp']]*10
        plan={'work_items':[{'submission_id':SUB,'icp_position':3,'scored_run_id':SOURCE,'output_ref':source['output_ref']}]}
        refs=p.service._scoring_work_items(p.store.get_round(ROUND),2,plan)[0]['company_judgment_refs'];p.refs=refs
        miss=lambda ref:{**{k:ref[k]for k in ('company_index','cache_key','company_input_hash')},'authority_slot':0}
        completed=[{**miss(ref),'raw_judgment':{**_raw(0),'company_judgment_policy':'company_judgment_v1','failure_reason':'required_intent_unverified'}}for ref in refs[:4]]
        partial=scoring.build_scoring_failure(SOURCE,'judge_error',company_judgments=completed)
        p.objects['b66-partial.json']=contracts.canonical_json(partial).encode();old_runner=hotkey('old-b66-judge639');exclusions=[old_runner]
        with p.conn.cursor()as q:
            q.execute('SET LOCAL session_replication_role=replica')
            for attempt in range(1,6):
                result={'terminal_status':'judge_error','failure_diagnostic':{'stage':'scorer','error_class':'judge_error','reason':'unknown'}}
                if attempt==1:result['company_judgment_completion_request_hash']='sha256:'+'f'*64
                claim={'company_judgment_cache':{'schema_version':company_judgments.LEASE_SCHEMA_VERSION,'hits':[],'misses':[miss(x)for x in refs]},'runner_authority_exclusions':exclusions}
                q.execute("UPDATE public.lab_arena_runs SET terminal_cause='judge_error',runner_hotkey=%s,company_judgment_refs=%s::jsonb,result_doc=%s::jsonb,claim_response=%s::jsonb,claim_request_id=%s,claim_request_hash=%s,lease_generation=1,output_ref=%s WHERE run_id=%s",(old_runner,json.dumps(refs),json.dumps(result),json.dumps(claim),'c'*31+str(attempt),'sha256:'+'d'*64,'b66-partial.json'if attempt==1 else None,ASSIGNMENT+':'+str(attempt)))
            for sequence in range(315):
                rid=ASSIGNMENT+':'+str(1+sequence//63);identity='sha256:'+hashlib.sha256(('old639:'+str(sequence)).encode()).hexdigest()
                for kind,doc,terminal in [('reservation',{'action_sequence':1+sequence%63,'request_hash':'sha256:'+'e'*64},None),('dispatch',{},None),('settlement',{}, {'status':200,'headers':{},'body_b64':'e30=','call_succeeded':True})]:
                    q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,round_id,run_id,submission_id,miner_hotkey,stage,provider,operation_id,funding_source,call_identity,amount_microusd,entry_doc,terminal_response) VALUES(%s,%s,%s,%s,%s,2,'openrouter','openrouter.chat','host',%s,1,%s::jsonb,%s::jsonb)",(kind,ROUND,rid,SUB,source['miner_hotkey'],identity,json.dumps(doc),json.dumps(terminal)if terminal else None))
            for new,ref in zip(completed,refs):
                evidence=company_judgments.build_evidence_snapshot(new_judgment=new,company_ref=ref,source_score_run_id=ASSIGNMENT+':1',source_scored_run_id=SOURCE,source_output_ref='b66-partial.json',source_output_hash=contracts.document_hash(partial),source_runner_hotkey=old_runner,source_claim_request_id='c'*31+'1',source_claim_request_hash='sha256:'+'d'*64,source_lease_generation=1,source_completion_request_hash='sha256:'+'f'*64,runner_authority_exclusions=exclusions)
                q.execute('INSERT INTO public.lab_arena_company_judgments(cache_key,authority_slot,scope_doc,company_input_hash,evidence_hash,evidence_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey) VALUES(%s,0,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s)',(ref['cache_key'],json.dumps(ref['scope_doc']),ref['company_input_hash'],contracts.document_hash(evidence),json.dumps(evidence),ASSIGNMENT+':1',SOURCE,old_runner))
            for i in range(3):
                key='sha256:'+str(7+i)*64
                q.execute("INSERT INTO public.lab_arena_signal_judgments(cache_key,authority_slot,signal_input_hash,scope_doc,judgment_hash,judgment_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey,source_claim_request_hash,source_lease_generation,runner_authority_exclusions) VALUES(%s,0,%s,%s::jsonb,%s,'{}',%s,%s,%s,%s,1,%s::jsonb)",(key,'sha256:'+'8'*64,json.dumps({'cache_key':key,'signal_input_hash':'sha256:'+'8'*64}),'sha256:'+'7'*64,ASSIGNMENT+':1',SOURCE,old_runner,'sha256:'+'d'*64,json.dumps(exclusions)))
            q.execute('DELETE FROM public.lab_arena_reviewer_runtimes')
            q.execute('INSERT INTO public.lab_arena_reviewer_runtimes VALUES(%s,%s,%s,%s,now())',(GRADING,REPOSITORY+'@'+GRADING,RUNTIME,REPOSITORY+'@'+RUNTIME))
            q.execute('SET LOCAL session_replication_role=origin')
        p.conn.commit();p.original_accepted={k:v for k,v in recovery.snapshot(p)['runs'].items()if v['status']=='accepted'};assert len(p.original_accepted)==333
        p.sql=recovery.render(p);recovery.apply(p)
        assert (p.store.get_round(ROUND)['status_generation'],p.store.get_round(ROUND)['stage_generation'])==(9,7)
        p.window_sql=window.render(p);window.apply(p)
        assert p.store.get_round(ROUND)['configuration_doc']['schedule']['final_scoring_close']=='2099-11-06T08:30:00Z'
        for _ in range(50):
            lease=recovery.claim(p);assert lease['status']=='leased'and lease['kind']=='execute',lease
            recovery.bill(p,lease);assert recovery.complete(p,lease)['status']=='accepted'
        assert p.service.close_stage(ROUND,2)['status']=='ok'
        row=p.store.get_round(ROUND);assert len(row['stage2_scoring_plan_doc']['work_items'])==215
        items=p.service._scoring_work_items(row,2,row['stage2_scoring_plan_doc'])
        assert p.store.open_scoring(ROUND,2,items,company_quality_cache=True)['status']=='ok'
        row=p.store.get_round(ROUND);assert (row['status'],row['status_generation'],row['stage_generation'])==('stage2_scoring',11,9),row
        for _ in range(50):
            lease=recovery.claim(p);assert lease['status']=='leased'and lease['kind']=='score',lease
            assert recovery.complete(p,lease)['status']=='accepted'
        p.stage_generations=(row['status_generation'],row['stage_generation']);p.sql=render(p)
        with p.conn.cursor()as q:
            assert scalar(q,"SELECT count(DISTINCT call_identity) FROM public.lab_arena_ledger WHERE entry_kind='settlement' AND run_id LIKE %s",(ASSIGNMENT+':%',))==315
            assert scalar(q,'SELECT public.lab_arena__judge_calls_clear_v1(%s)',(ASSIGNMENT,))is True
        yield p
    finally:p.store.close();p.conn.close()


def render(p):
    with p.conn.cursor()as q:
        hashes=lambda table,where:scalar(q,'SELECT jsonb_object_agg('+('run_id'if table=='lab_arena_runs'else'cache_key')+",encode(extensions.digest(to_jsonb(a)::text,'sha256'),'hex')) FROM public."+table+' a WHERE '+where)
        manifest={'source_sha256':hashes('lab_arena_runs',"run_id='"+SOURCE+"'")[SOURCE],'prior_run_hashes':hashes('lab_arena_runs',"assignment_id='"+ASSIGNMENT+"'"),'company_judgment_refs':p.refs,'runtime_image_digest':RUNTIME,'runtime_image_reference':REPOSITORY+'@'+RUNTIME}
        for table in ('lab_arena_company_judgments','lab_arena_signal_judgments'):manifest[table+'_row_hashes']=hashes(table,"source_scored_run_id='"+SOURCE+"'")
        cfg=p.store.get_round(ROUND)['configuration_doc'];digest=recovery.source.hash_doc(q,cfg)
    text=re.sub(r'\$manifest639\$.*?\$manifest639\$',lambda _:'$manifest639$'+json.dumps(manifest,separators=(',',':'))+'$manifest639$',SQL.read_text(),flags=re.S)
    return text.replace('69befd3303a62b7ef7ceb28799f2082c83304b281884ddc6c3de693ba19b6e15',digest).replace('2026-11-06T08:30:00Z','2099-11-06T08:30:00Z').replace('2026-11-06T08:35:00Z','2099-11-06T08:35:00Z')


@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
@pytest.mark.parametrize('phase',['scoring','judged'])
def test_full_recovery_then_exact_sixth_claim_reuses_four_without_seventh(prepared,outcome,phase):
    p=prepared
    if phase=='judged':
        # The ordinary driver closes scoring early when every existing score
        # row is terminal, even while exhausted judgments remain unfinished.
        assert p.service.close_scoring(ROUND,2)['status']=='closed'
        row=p.store.get_round(ROUND)
        assert (row['status'],row['status_generation'],row['stage_generation'])==('stage2_judged',12,10)
    if phase=='judged'and outcome=='accepted':
        # Isolate this pending sixth as the only unfinished judgment. Other
        # exhausted assignments become accepted only in this disposable control.
        with p.conn.cursor()as q:
            rows=scalar(q,"SELECT jsonb_agg(to_jsonb(a)) FROM (SELECT DISTINCT ON(assignment_id) * FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND assignment_id<>%s ORDER BY assignment_id,attempt DESC)a WHERE status='failed'",(ROUND,ASSIGNMENT))
            assert len(rows)==16
            q.execute('SET LOCAL session_replication_role=replica')
            for r in rows:
                ref=r['run_id']+'-accepted-control.json'
                p.objects[ref]=contracts.canonical_json({'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':r['scored_run_id'],'breakdowns':[],'company_judgments':[]}).encode()
                q.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',result_doc=%s::jsonb,output_ref=%s WHERE run_id=%s",(json.dumps({'terminal_status':'accepted','company_judgment_consumed_indexes':[]}),ref,r['run_id']))
            q.execute('SET LOCAL session_replication_role=origin')
        p.conn.commit()
    before=snapshot(p);recovery.apply(p);recovery.apply(p)
    after=snapshot(p)
    assert {k:v for k,v in after['runs'].items()if k!=RUN}==before['runs']
    assert {k:v for k,v in after.items()if k!='runs'}=={k:v for k,v in before.items()if k!='runs'}
    assert all(after['runs'][k]==v for k,v in p.original_accepted.items())
    if phase=='judged'and outcome=='accepted':
        waiting=p.service.score_stage(ROUND,2)
        assert waiting['status']=='retry'and waiting['remaining']==1
        assert p.service.publish(ROUND)['status']=='stale'
        assert p.store.get_round(ROUND)['published_at']is None
    with p.conn.cursor()as q:
        assert scalar(q,'SELECT public.lab_arena__score_attempt_count_v1(a) FROM public.lab_arena_runs a WHERE run_id=%s',(RUN,))==6
        q.execute('SET LOCAL session_replication_role=replica');q.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s",(RUN,));q.execute('SET LOCAL session_replication_role=origin')
        post=scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(CLAIM,))
        pre=post.replace('\n      OR public.lab_arena__nov05_b66_judge639_v1(v_round,runs)','');assert pre!=post;q.execute(pre)
    p.conn.commit();assert claim(p)['status']!='leased'
    with p.conn.cursor()as q:q.execute(post)
    p.conn.commit();lease=claim(p)
    assert lease['status']=='leased'and lease['run_id']==RUN,lease
    assert sorted(r['company_index']for r in lease['company_judgment_cache']['hits'])==[0,1,2,3]
    assert [r['company_index']for r in lease['company_judgment_cache']['misses']]==[4]
    assert p.store.get_run(RUN)['reviewer_runtime_doc']['runtime_image_digest']==RUNTIME
    # No provider is contacted: exercise native funding, reservation,
    # dispatch and settlement for this exact sixth lease before completion.
    if outcome!='accepted':recovery.bill(p,lease)
    if outcome=='accepted':
        # Consume the four unchanged cached results and evaluate only index4.
        from lab_arena.service import ArenaService,ServiceConfig
        from tests.lab_arena.test_lab_arena_service_round import FakeChain,keypair,wallet_verify
        from tests.lab_arena.legacy_partial_postgres_test import _completion
        companies=_input()['companies'];attempted=[]
        def scorer(companies,icp,_reference):
            attempted.extend(c['company_name']for c in companies)
            assert attempted==['E']
            recovery.bill(p,lease)
            raw={**_raw(0),'company_judgment_policy':'company_judgment_v1','failure_reason':'required_intent_unverified'}
            indexes=scoring.verify.bucket_skip(icp,companies,max_scored_companies=0)[0]
            return scoring.project_company_judgments(companies,[raw],company_quality=False,scored_indexes=indexes)
        scorer.integrity_policy=True;scorer.company_quality=False;scorer.contacts_required=False
        breakdowns,new=scoring.score_quality_work_item({'scored_run_id':SOURCE},icp=_input()['icp'],companies=companies,scorer=scorer,cache_context=lease['company_judgment_cache'])
        assert attempted==['E'] and [r['company_index']for r in new]==[4]
        output={'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':SOURCE,'breakdowns':breakdowns,'company_judgments':new}
        key=keypair('633-runner')
        # Gateway validation creates the authoritative cache evidence from the
        # saved object. A signed completion also verifies unchanged cache hits.
        objects=type('Objects',(),{'get':lambda _,k:p.objects[k],'get_bounded':lambda _,k,n:p.objects[k],'put':lambda _,k,v:p.objects.__setitem__(k,v)})()
        gateway=ArenaService(ServiceConfig(mode='live',store=p.store,object_store=objects,signer=None,chain=FakeChain([key.ss58_address]),verify_signature=wallet_verify,daily_icp_source=lambda **_: {},banned_hotkeys_source=lambda:(),broker_factory=lambda *_:None))
        gateway.evaluation_icps=p.service.evaluation_icps
        # The fixture's SQL claim uses hotkey('633-runner'); sign with the
        # matching deterministic key used by the shared test helpers.
        current=p.store.get_run(RUN)
        assert key.ss58_address==current['runner_hotkey']
        request=_completion(ROUND,key,lease,'recovery633-fixture',output,terminal='accepted')
        assert gateway.handle_complete(request)['status']=='accepted'
        assert gateway.handle_complete(request)['idempotent'] is True
    elif outcome=='judge_error':assert recovery.complete(p,lease,cause='judge_error')['status']=='failed'
    else:
        with p.conn.cursor()as q:
            q.execute('SET LOCAL session_replication_role=replica');q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s",(RUN,));q.execute('SET LOCAL session_replication_role=origin')
        p.conn.commit()
    p.store.expire_leases(ROUND)
    assert max(r['attempt']for r in p.store.list_runs(ROUND,submission_id=SUB,kind='score')if r['assignment_id']==ASSIGNMENT)==6
    assert claim(p)['status']!='leased'
    terminal=snapshot(p);old_ledger={r['entry_id']:r for r in before['ledger']};new_ledger={r['entry_id']:r for r in terminal['ledger']}
    assert all(new_ledger[k]==v for k,v in old_ledger.items())
    assert len(new_ledger)==len(old_ledger)+3
    recovery.apply(p);assert snapshot(p)==terminal
    with p.conn.cursor()as q:
        for table in ('lab_arena_company_judgments','lab_arena_signal_judgments'):
            for old in before[table]:assert scalar(q,'SELECT to_jsonb(a) FROM public.'+table+' a WHERE cache_key=%s AND authority_slot=0',(old['cache_key'],))==old
        if outcome!='accepted':
            # Even an externally inserted seventh is outside the one-row
            # allowance. Native claiming must reject it at the ordinary cap.
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,stage_generation,kind,scored_run_id,company_judgment_refs,scoring_generation_floor,judgment_group_leader,created_at) SELECT assignment_id||':7',assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,7,'pending',stage_generation,kind,scored_run_id,company_judgment_refs,0,NULL,now()-interval '30 minutes' FROM public.lab_arena_runs WHERE run_id=%s",(RUN,))
            q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    if outcome!='accepted':assert claim(p)['status']!='leased'
    else:
        with p.conn.cursor()as q:
            history=scalar(q,'SELECT public.lab_arena_scoring_policy_history_v1(%s)',(ROUND,))
            revision=next(r for r in history['revisions']if(r.get('recovery')or{}).get('operation')=='nov05_b66_post_repair639_v1')
            assert revision['generation_floor']==0 and revision['receipt_generation']>0
        if phase=='scoring':assert p.service.close_scoring(ROUND,2)['status']=='closed'
        if phase=='judged':
            # Completing the sole pending sixth releases the real assembly
            # gate; no deadline move is needed in this isolated control.
            assert p.service.score_stage(ROUND,2)['status']=='ok'
            waiting=p.service.publish(ROUND)
            assert waiting['status']=='retry'and waiting['reason']=='execution_infrastructure_incomplete'
        else:assert p.service.score_stage(ROUND,2)['status']=='retry'
        # Only the disposable clock/schedule move past the publication gate.
        # The normal scorer still excludes genuinely unfinished assignments.
        now=datetime.now(timezone.utc)
        with p.conn.cursor()as q:
            q.execute('SET LOCAL session_replication_role=replica')
            cfg=p.store.get_round(ROUND)['configuration_doc']
            cfg['schedule'].update(stage_2_close=(now-timedelta(seconds=3)).isoformat(),final_scoring_close=(now-timedelta(seconds=2)).isoformat(),publication_deadline=(now-timedelta(seconds=1)).isoformat())
            q.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',(json.dumps(cfg),ROUND))
            q.execute('SET LOCAL session_replication_role=origin')
        p.conn.commit();p.service._clock=lambda:now
        if phase=='scoring':assert p.service.score_stage(ROUND,2)['status']=='ok'
        assert p.service.publish(ROUND)['status']=='ok'
        final=p.store.get_round(ROUND)
        assert final['status']=='published'and final['scoring_generation_floor']==0
        assert p.store.get_run(RUN)['status']=='accepted'


@pytest.mark.parametrize('fault',['source','parent','company','signal','runtime','hold','inflight','plan','expired','stage_generation','crossed_scoring_generation','crossed_judged_generation','later_judged_generation','wrong_judged_stage'])
def test_changed_binding_refuses_exact_sixth_and_rolls_back(prepared,fault):
    p=prepared
    with p.conn.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        if fault=='source':q.execute("UPDATE public.lab_arena_runs SET output_ref='changed.json' WHERE run_id=%s",(SOURCE,))
        elif fault=='parent':q.execute("UPDATE public.lab_arena_runs SET result_doc='{}' WHERE run_id=%s",(ASSIGNMENT+':5',))
        elif fault=='company':q.execute("UPDATE public.lab_arena_company_judgments SET evidence_hash='sha256:'||repeat('0',64)")
        elif fault=='signal':q.execute("UPDATE public.lab_arena_signal_judgments SET judgment_hash='sha256:'||repeat('0',64)")
        elif fault=='runtime':q.execute("UPDATE public.lab_arena_reviewer_runtimes SET runtime_image_digest='sha256:'||repeat('0',64),runtime_image_reference=%s||'@sha256:'||repeat('0',64)",(REPOSITORY,))
        elif fault=='hold':q.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton')
        elif fault=='inflight':q.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,run_id,call_identity,amount_microusd,entry_doc) SELECT 'reservation',miner_hotkey,round_id,run_id,%s,1,'{}' FROM public.lab_arena_runs WHERE run_id=%s",('sha256:'+'e'*64,ASSIGNMENT+':5'))
        elif fault=='plan':q.execute('UPDATE public.lab_arena_rounds SET stage2_scoring_plan_doc=NULL WHERE round_id=%s',(ROUND,))
        elif fault=='expired':q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((now()-interval '1 second')::text)) WHERE round_id=%s",(ROUND,))
        elif fault=='stage_generation':q.execute('UPDATE public.lab_arena_rounds SET stage_generation=stage_generation+1 WHERE round_id=%s',(ROUND,))
        elif fault=='crossed_scoring_generation':q.execute('UPDATE public.lab_arena_rounds SET status_generation=12 WHERE round_id=%s',(ROUND,))
        elif fault=='crossed_judged_generation':q.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged',status_generation=11 WHERE round_id=%s",(ROUND,))
        elif fault=='wrong_judged_stage':q.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged',status_generation=12 WHERE round_id=%s",(ROUND,))
        elif fault=='later_judged_generation':q.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged',status_generation=13 WHERE round_id=%s",(ROUND,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit();before=snapshot(p)
    with pytest.raises(p.pg.Error):recovery.apply(p)
    p.conn.rollback();assert snapshot(p)==before
