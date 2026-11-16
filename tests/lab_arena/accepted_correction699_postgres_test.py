"""Accepted ICP repair uses one finite native child and exact cache retirement.

Only disposable fixture setup seeds historical accepted rows. Recovery, claims,
expiry, accounting and score writes use real PostgreSQL security and RPCs.
"""
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.lab_arena import finite_child_scorer_correction679_postgres_test as f

old, base, prior = f.old, f.base, f.prior
for name in ('database','prepared','prepared_company','case','base_extended','extended','provider_case','plain_finite','before','post','prior_bug','budget_parent','corrected'):
    globals()[name] = getattr(f, name)
SQL = Path(__file__).parents[2] / 'scripts/699-lab-arena-targeted-accepted-judgment-correction.sql'
CALL = 'lab_arena_correct_accepted_judgment_v1(%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s)'
SIGNATURE = 'public.lab_arena_correct_accepted_judgment_v1(text,text,text,text,jsonb,jsonb,jsonb,text,text,text,boolean)'
SIGNATURES = [
    'public.lab_arena_runs_terminal_v1()',
    'public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)',
    'public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs)',
    'public.lab_arena__operator_scorer_bug_claim_v1(public.lab_arena_rounds,public.lab_arena_runs)',
    'public.lab_arena__prepare_company_judgment_claim_v1(text,text,bigint,timestamptz)',
    'public.lab_arena_record_run_scores(text,smallint,jsonb)',
]


def row_hash(q, run_id):
    return old.scalar(q,"SELECT 'sha256:'||encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') FROM lab_arena_runs r WHERE run_id=%s", (run_id,))


@pytest.fixture
def accepted(corrected):
    p = corrected
    with p.c.cursor() as q:
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE lab_arena_runs SET status='accepted',terminal_cause='accepted',result_doc='{" + '"terminal_status":"accepted"' + "}'::jsonb,output_ref='accepted-judge-fixture' WHERE run_id=%s",(p.parent['run_id'],))
        q.execute("UPDATE lab_arena_runs SET per_icp_score=12,qualification_doc='{"+'"companies":[]'+"}'::jsonb WHERE run_id=%s",(p.parent['scored_run_id'],))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
        p.parent = old.scalar(q,'SELECT to_jsonb(r) FROM lab_arena_runs r WHERE run_id=%s',(p.parent['run_id'],))
        p.parent_hash=row_hash(q,p.parent['run_id']);p.source_hash=row_hash(q,p.parent['scored_run_id'])
        p.checkpoints=old.scalar(q,"SELECT jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')) FROM lab_arena_company_judgments j WHERE EXISTS(SELECT 1 FROM jsonb_array_elements(%s::jsonb)r WHERE r->>'cache_key'=j.cache_key)",(json.dumps(p.parent['company_judgment_refs']),))
        p.signal_checkpoints={}
        ref=p.parent['company_judgment_refs'][0]
        p.affected=[{k:ref[k] for k in ('company_index','cache_key','company_input_hash')} | {'authority_slot':0,'evidence_hash':old.scalar(q,'SELECT evidence_hash FROM lab_arena_company_judgments WHERE cache_key=%s AND authority_slot=0',(ref['cache_key'],))}]
    p.c.commit()
    return p


@pytest.fixture
def installed(accepted):
    yield from installation(accepted)


@pytest.fixture
def existing_flow(corrected):
    yield from installation(corrected)


def installation(p):
    with p.c.cursor() as q:
        originals=[old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(s,)) for s in SIGNATURES]
        q.execute(SQL.read_text())
    p.c.commit()
    try:
        yield p
    finally:
        p.c.rollback()
        with p.c.cursor() as q:
            q.execute('DROP FUNCTION '+SIGNATURE)
            q.execute('DROP FUNCTION public.lab_arena_retired_accepted_judgment_v1(TEXT)')
            for d in originals:q.execute(d)
        p.c.commit()


@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
def test_existing_finite_corrections_still_use_the_same_native_flow(existing_flow,outcome):
    f.test_one_reviewed_correction_retains_work_payer_and_native_finality(existing_flow,outcome)


def args(p, **changes):
    values=dict(parent=p.parent['run_id'],parent_hash=p.parent_hash,source_hash=p.source_hash,runtime=f.RUNTIME,
                affected=p.affected,checkpoints=p.checkpoints,signals=p.signal_checkpoints,
                request='accepted-correction-fixture',actor='fixture-owner',correction='git:proved-citation-defect',dry=False)
    values.update(changes)
    return values


def grant(p, **changes):
    values=args(p,**changes)
    for k in ('affected','checkpoints','signals'):values[k]=json.dumps(values[k])
    with p.c.cursor() as q:result=old.scalar(q,'SELECT public.'+CALL,tuple(values.values()))
    p.c.commit()
    return result


def claim_child(p, child):
    with p.c.cursor() as q:
        q.execute("UPDATE lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s", (child,))
    p.c.commit();old.release_guard(p)
    lease=old.claim(p)
    assert lease.get('run_id')==child,lease
    return lease


def test_accepted_parent_not_supported_by_failed_recovery(accepted):
    p=accepted;before=base.snapshot(p)
    with pytest.raises(p.pg.Error):base.grant(p,runtime=f.RUNTIME)
    p.c.rollback();assert base.snapshot(p)==before


def test_dry_run_exact_retirement_child_and_checkpoint_reuse(installed):
    p=installed;old.release_guard(p);before=base.snapshot(p)
    preview=grant(p,dry=True);assert base.snapshot(p)==before
    result=grant(p);assert result['child_identity']==preview['child_identity']
    after=base.snapshot(p)
    assert len(after['lab_arena_runs'])==len(before['lab_arena_runs'])+1
    assert len(after['lab_arena_scoring_supersessions'])==len(before['lab_arena_scoring_supersessions'])+1
    oldruns={r['run_id']:r for r in before['lab_arena_runs']};newruns={r['run_id']:r for r in after['lab_arena_runs']}
    for run_id,row in oldruns.items():
        if run_id==p.parent['run_id']:
            excluded={'status','terminal_cause','terminal_doc','updated_at'}
            assert newruns[run_id]['status']=='failed' and newruns[run_id]['terminal_cause']=='stage_closed'
        elif run_id==p.parent['scored_run_id']:
            excluded={'per_icp_score','qualification_doc','updated_at'}
            assert newruns[run_id]['per_icp_score'] is None
        else:excluded=set()
        assert {k:v for k,v in row.items() if k not in excluded}=={k:v for k,v in newruns[run_id].items() if k not in excluded}
    for table in set(before)-{'lab_arena_runs','lab_arena_scoring_supersessions'}:assert before[table]==after[table]
    assert grant(p)['idempotent'] is True and base.snapshot(p)==after
    child=result['child_identity']['run_id'];assert old.funding(p,child)['funding_source']==('host' if p.organizer else 'miner_key')
    assert old.funding(p,child,'deepline')['funding_source']=='miner_key'
    lease=claim_child(p,child)
    cache=lease['company_judgment_cache']
    assert len(cache['hits'])==1 and cache['hits'][0]['company_index']==1
    assert len(cache['misses'])==1 and cache['misses'][0]['company_index']==0
    assert cache['misses'][0]['authority_slot']==1
    assert cache['hits'][0]['authority_slot']==0


@pytest.mark.parametrize('fault',['hash','source_hash','runtime','same_runtime','affected_evidence','affected_index',
                                  'affected_slot','duplicate','checkpoints','signals','expired','published','rewarded','later','frozen_plan','calls_pending','wrong_phase','frozen_source'])
def test_invalid_correction_is_atomic(installed,fault):
    p=installed;changes={}
    if fault=='hash':changes['parent_hash']='sha256:'+'0'*64
    elif fault=='source_hash':changes['source_hash']='sha256:'+'0'*64
    elif fault=='runtime':changes['runtime']='sha256:'+'a'*64
    elif fault=='same_runtime':prior.mapping(p,p.parent['reviewer_runtime_doc']['runtime_image_digest']);changes['runtime']=p.parent['reviewer_runtime_doc']['runtime_image_digest']
    elif fault.startswith('affected_'):
        a=copy.deepcopy(p.affected);a[0][{'affected_evidence':'evidence_hash','affected_index':'company_index','affected_slot':'authority_slot'}[fault]]='sha256:'+'0'*64 if fault=='affected_evidence' else 99;changes['affected']=a
    elif fault=='duplicate':changes['affected']=p.affected*2
    elif fault=='checkpoints':changes['checkpoints']={}
    elif fault=='signals':changes['signals']={'bogus':'hash'}
    else:
        with p.c.cursor() as q:
            q.execute('SET LOCAL session_replication_role=replica')
            if fault=='expired':q.execute("UPDATE lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}','\"2000-01-01T00:00:00Z\"'::jsonb) WHERE round_id=%s",(old.ROUND,))
            elif fault=='published':q.execute("UPDATE lab_arena_rounds SET status='published',published_at=clock_timestamp() WHERE round_id=%s",(old.ROUND,))
            elif fault=='rewarded':q.execute("UPDATE lab_arena_rounds SET reward_activated_at=clock_timestamp() WHERE round_id=%s",(old.ROUND,))
            elif fault=='frozen_plan':q.execute("UPDATE lab_arena_rounds SET stage2_scoring_plan_doc='{}' WHERE round_id=%s",(old.ROUND,))
            elif fault=='wrong_phase':q.execute("UPDATE lab_arena_rounds SET status='open' WHERE round_id=%s",(old.ROUND,))
            elif fault=='frozen_source':
                q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
                q.execute("UPDATE lab_arena_runs SET judgment_input_hash='sha256:'||repeat('0',64) WHERE run_id=%s",(p.parent['run_id'],))
                q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
                changes['parent_hash']=row_hash(q,p.parent['run_id'])
            elif fault=='calls_pending':old.insert(q,'lab_arena_ledger',dict(
                entry_kind='reservation',run_id=p.parent['run_id'],round_id=old.ROUND,
                submission_id=p.parent['submission_id'],miner_hotkey=p.parent['miner_hotkey'],
                stage=2,call_identity='sha256:'+'7'*64,provider='openrouter',operation_id='openrouter.chat',
                funding_source='host' if p.organizer else 'miner_key',amount_microusd=500,entry_doc={}))
            elif fault=='later':
                row={k:p.parent[k] for k in ('assignment_id','round_id','submission_id','miner_hotkey','stage','icp_position','scored_run_id','company_judgment_refs','stage_generation','scoring_generation_floor')}
                row.update(run_id=p.parent['assignment_id']+':99',attempt=99,status='failed',kind='score',terminal_cause='judge_error')
                old.insert(q,'lab_arena_runs',row)
            q.execute('SET LOCAL session_replication_role=origin')
        p.c.commit()
    before=base.snapshot(p)
    for dry in (True,False):
        with pytest.raises(p.pg.Error):grant(p,dry=dry,**changes)
        p.c.rollback();assert base.snapshot(p)==before


@pytest.mark.parametrize('outcome',['judge_error','expired'])
def test_finite_child_cannot_loop_or_be_corrected_twice(installed,outcome):
    p=installed;child=grant(p)['child_identity']['run_id'];claim_child(p,child)
    if outcome=='expired':
        with p.c.cursor() as q:q.execute("UPDATE lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(child,))
        p.c.commit()
    else:prior.complete_failed(p,child)
    old.rpc(p,'lab_arena_expire_leases(%s)',(old.ROUND,))
    with p.c.cursor() as q:
        assert old.scalar(q,"SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s",(p.parent['assignment_id'],p.parent['attempt']+1))==0
    p.c.commit();before=base.snapshot(p)
    with pytest.raises(p.pg.Error):grant(p,request='second-correction')
    p.c.rollback();assert base.snapshot(p)==before


def test_owner_only_and_idempotent_migration(installed):
    p=installed;before=base.snapshot(p)
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
        for role in ('anon','authenticated','service_role','lab_arena_service'):
            assert old.scalar(q,'SELECT has_function_privilege(%s,%s,\'EXECUTE\')',(role,SIGNATURE)) is False
    p.c.commit();assert base.snapshot(p)==before


@pytest.mark.parametrize('fault',['hash','terminal','source'])
def test_retired_source_proof_rejects_tampering(installed,fault):
    p=installed;grant(p)
    with p.c.cursor() as q:
        q.execute('SAVEPOINT before_tampering')
        q.execute('SET LOCAL session_replication_role=replica')
        if fault=='hash':
            q.execute("UPDATE lab_arena_scoring_supersessions SET request_hash='sha256:'||repeat('0',64) WHERE request_id='accepted-correction-fixture'")
        else:
            q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
            field,value=('terminal_doc',json.dumps({'operation':'wrong'})) if fault=='terminal' else ('output_ref','changed-output')
            q.execute('UPDATE lab_arena_runs SET '+field+'=%s'+('::jsonb' if fault=='terminal' else '')+' WHERE run_id=%s',(value,p.parent['run_id']))
        assert old.scalar(q,'SELECT lab_arena_retired_accepted_judgment_v1(%s)',(p.parent['run_id'],)) is False
        q.execute('ROLLBACK TO SAVEPOINT before_tampering')
        assert old.scalar(q,'SELECT lab_arena_retired_accepted_judgment_v1(%s)',(p.parent['run_id'],)) is True
    p.c.commit()


def test_two_concurrent_grants_create_one_child(installed):
    p=installed
    def invoke(request):
        connection=p.pg.connect(**p.dsn);local=SimpleNamespace(**{**p.__dict__,'c':connection})
        try:
            return grant(local,request=request)
        except p.pg.Error:
            connection.rollback();return None
        finally:connection.close()
    with ThreadPoolExecutor(2) as executor:results=list(executor.map(invoke,['race-a','race-b']))
    assert sum(r is not None for r in results)==1
    with p.c.cursor() as q:assert old.scalar(q,"SELECT count(*) FROM lab_arena_runs WHERE assignment_id=%s AND attempt>%s",(p.parent['assignment_id'],p.parent['attempt']))==1
    p.c.commit()


def write_projection(p, score_run_id, score=23):
    item={'run_id':p.parent['scored_run_id'],'per_icp_score':score,
          'score_run_id':score_run_id,'scoring_generation_floor':p.parent['scoring_generation_floor'],
          'qualification_doc':{'companies':[]}}
    return old.rpc(p,'lab_arena_record_run_scores(%s,%s::smallint,%s::jsonb)',(old.ROUND,2,json.dumps([item])))


def projection_phase(p):
    # Only fixture setup enters the normal score-projection phase.
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s",(old.ROUND,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()


def complete_corrected_company(p, lease):
    from lab_arena import company_judgments, contracts, scoring
    from lab_arena.store import ArenaStore, PsycopgTransport
    miss=lease['company_judgment_cache']['misses'][0]
    with p.c.cursor() as q:
        raw=old.scalar(q,"SELECT evidence_doc->'raw_judgment' FROM lab_arena_company_judgments WHERE cache_key=%s AND authority_slot=0",(miss['cache_key'],))
        source=old.scalar(q,'SELECT to_jsonb(r) FROM lab_arena_runs r WHERE run_id=%s',(lease['run_id'],))
    p.c.commit()
    output=scoring.build_scoring_output(lease['scored_run_id'],[],company_judgments=[{**miss,'raw_judgment':raw}])
    evidence=company_judgments.build_evidence_snapshot(
        new_judgment={**miss,'raw_judgment':raw},company_ref=p.parent['company_judgment_refs'][0],
        source_score_run_id=lease['run_id'],source_scored_run_id=lease['scored_run_id'],
        source_output_ref=lease['run_id']+'-accepted-output',source_output_hash=contracts.document_hash(output),
        source_runner_hotkey=source['runner_hotkey'],source_claim_request_id=source['claim_request_id'],
        source_claim_request_hash=source['claim_request_hash'],source_lease_generation=source['lease_generation'],
        source_completion_request_hash='sha256:'+'e'*64,runner_authority_exclusions=lease['runner_authority_exclusions'])
    transport=PsycopgTransport(lambda:p.pg.connect(**p.dsn))
    try:
        return ArenaStore(transport).complete_attempt(
            run_id=lease['run_id'],lease_token_hash=old.TOKEN,result={'terminal_status':'accepted'},
            terminal_cause='accepted',output_ref=evidence['source_output_ref'],output_hash=contracts.document_hash(output),
            company_judgment_evidence=[{k:evidence[k] for k in ('cache_key','company_input_hash','authority_slot')}
                | {'evidence_hash':contracts.document_hash(evidence),'evidence_doc':evidence}],
            completion_request_hash='sha256:'+'e'*64)
    finally:transport.close()


def test_completion_restores_only_new_bound_score_and_keeps_original(installed):
    p=installed;projection_phase(p);result=grant(p);child=result['child_identity']['run_id']
    with p.c.cursor() as q:
        assert old.scalar(q,'SELECT lab_arena_retired_accepted_judgment_v1(%s)',(p.parent['run_id'],)) is True
        assert old.scalar(q,'SELECT lab_arena_retired_accepted_judgment_v1(%s)',(child,)) is False
    p.c.commit()

    for run_id in (None,p.parent['run_id'],child):
        with pytest.raises(p.pg.Error,match='corrected_score_authority'):write_projection(p,run_id)
        p.c.rollback()
    lease=claim_child(p,child)
    result=complete_corrected_company(p,lease)
    assert result['status']=='accepted'
    assert result['company_judgments_stored']==1
    for run_id in (None,p.parent['run_id']):
        with pytest.raises(p.pg.Error,match='corrected_score_authority'):write_projection(p,run_id)
        p.c.rollback()
    assert write_projection(p,child)['recorded']==1
    assert write_projection(p,child)['existing']==1
    with p.c.cursor() as q:
        assert old.scalar(q,'SELECT per_icp_score FROM lab_arena_runs WHERE run_id=%s',(p.parent['scored_run_id'],))==23
        archived=old.scalar(q,"SELECT recovery_doc->'superseded_judgment' FROM lab_arena_scoring_supersessions WHERE request_id=%s",('accepted-correction-fixture',))
        assert archived==p.parent
    p.c.commit()


def test_stale_projection_waiting_on_control_lock_cannot_restore_retired_score(installed):
    p=installed;projection_phase(p)
    with p.c.cursor() as q:
        q.execute("SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0))")

    def stale_writer():
        local=SimpleNamespace(**{**p.__dict__,'dsn':{
            **p.dsn,'application_name':'accepted699-stale-projection',
            'options':'-c statement_timeout=8000'}})
        try:
            write_projection(local,p.parent['run_id'],score=12)
            return 'unexpected-write'
        except p.pg.Error as exc:
            return str(exc)

    with ThreadPoolExecutor(1) as executor:
        future=executor.submit(stale_writer)
        deadline=time.monotonic()+5
        while True:
            with p.c.cursor() as q:
                q.execute('SELECT pg_stat_clear_snapshot()')
                waiting=old.scalar(q,"SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE application_name='accepted699-stale-projection' AND wait_event='advisory')")
            if waiting:break
            assert time.monotonic()<deadline,'stale writer never reached the shared control lock'
            time.sleep(.01)
        grant(p)
        assert 'corrected_score_authority' in future.result(timeout=5)
    with p.c.cursor() as q:
        assert old.scalar(q,'SELECT per_icp_score FROM lab_arena_runs WHERE run_id=%s',(p.parent['scored_run_id'],)) is None
    p.c.commit()


def test_baseline_uses_the_same_targeted_cache_correction_and_native_completion(installed):
    p=installed
    with p.c.cursor() as q:
        round_row=old.scalar(q,'SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s',(old.ROUND,))
        baseline=old.ROUND.replace('arena-','baseline-',1)
        hotkey=round_row['configuration_doc']['baseline_hotkey']
        source=old.scalar(q,'SELECT to_jsonb(r) FROM lab_arena_runs r WHERE run_id=%s',(p.parent['scored_run_id'],))
        aid=old.ROUND+':'+baseline+':1:0'
        source.update(run_id=aid+':1',assignment_id=aid,submission_id=baseline,miner_hotkey=hotkey,stage=1,attempt=1)
        parent={**p.parent,'run_id':aid+':score:1','assignment_id':aid+':score','submission_id':baseline,
                'miner_hotkey':hotkey,'stage':1,'attempt':1,'scored_run_id':source['run_id'],
                'claim_request_id':'9'*32}
        q.execute('SET LOCAL session_replication_role=replica')
        old.insert(q,'lab_arena_submissions',dict(submission_id=baseline,miner_hotkey=hotkey,round_id=old.ROUND,
            is_king=True,status='frozen',source_ref='arena/'+old.ROUND+'/sources/'+baseline+'.tar.gz'))
        q.execute('ALTER TABLE lab_arena_runs DISABLE TRIGGER lab_arena_runs_reviewer_runtime')
        old.insert(q,'lab_arena_runs',source);old.insert(q,'lab_arena_runs',parent)
        q.execute('ALTER TABLE lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_reviewer_runtime')
        q.execute("UPDATE lab_arena_rounds SET participants=participants||%s::jsonb,stage1_scoring_plan_doc=%s::jsonb WHERE round_id=%s",(
            json.dumps([{'submission_id':baseline,'miner_hotkey':hotkey,'is_king':True}]),
            json.dumps({'work_items':[{'scored_run_id':source['run_id'],'submission_id':baseline,
                                      'icp_position':0,'output_ref':source['output_ref']}]}),old.ROUND))
        q.execute('SET LOCAL session_replication_role=origin')
        p.parent=old.scalar(q,'SELECT to_jsonb(r) FROM lab_arena_runs r WHERE run_id=%s',(parent['run_id'],))
        p.parent_hash=row_hash(q,parent['run_id']);p.source_hash=row_hash(q,source['run_id'])
    p.c.commit()
    child=grant(p)['child_identity']['run_id']
    lease=claim_child(p,child)
    assert lease['stage']==1
    assert len(lease['company_judgment_cache']['hits'])==1
    assert len(lease['company_judgment_cache']['misses'])==1
    result=complete_corrected_company(p,lease)
    assert result['status']=='accepted' and result['company_judgments_stored']==1


@pytest.mark.parametrize('target',SIGNATURES+[SIGNATURE,'public.lab_arena_retired_accepted_judgment_v1(text)'])
@pytest.mark.parametrize('fault',['body','acl'])
def test_migration_reapply_rejects_drift(installed,target,fault):
    p=installed
    with p.c.cursor() as q:
        original=old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(target,))
        acl=old.scalar(q,'SELECT proacl::text FROM pg_proc WHERE oid=%s::regprocedure',(target,))
        if fault=='body':q.execute(original.replace('$function$\n','$function$\n-- unexpected body drift\n',1))
        else:q.execute('GRANT EXECUTE ON FUNCTION '+target+' TO PUBLIC')
    p.c.commit()
    try:
        with p.c.cursor() as q:
            with pytest.raises(p.pg.Error,match='correction699'):q.execute(SQL.read_text())
        p.c.rollback()
    finally:
        with p.c.cursor() as q:
            q.execute(original)
            q.execute('REVOKE EXECUTE ON FUNCTION '+target+' FROM PUBLIC')
            assert old.scalar(q,'SELECT proacl::text FROM pg_proc WHERE oid=%s::regprocedure',(target,))==acl
        p.c.commit()


def test_migration_reapply_rejects_changed_dry_run_default(installed):
    p=installed
    with p.c.cursor() as q:
        original=old.scalar(q,'SELECT pg_get_functiondef(%s::regprocedure)',(SIGNATURE,))
        assert original.count('DEFAULT true')==1
        q.execute(original.replace('DEFAULT true','DEFAULT false'))
    p.c.commit()
    try:
        with p.c.cursor() as q:
            with pytest.raises(p.pg.Error,match='correction699'):q.execute(SQL.read_text())
        p.c.rollback()
    finally:
        with p.c.cursor() as q:q.execute(original)
        p.c.commit()
