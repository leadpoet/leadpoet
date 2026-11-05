"""The exact infrastructure successor preserves clean work and uses native leases."""
import json
import base64
import time
import queue
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path

import pytest

from lab_arena import contracts, scoring, verify
from lab_arena.store import hash_lease_token
from tests.lab_arena import nov04_baseline_source621_postgres_test as prior

# Reuse the exact deployed621 disposable PostgreSQL schema.
@pytest.fixture
def database():
    generator=prior.database.__wrapped__()
    try:yield next(generator)
    finally:generator.close()
prepared = prior.prepared
ROUND = prior.ROUND
BASELINE = prior.BASELINE
SQL = prior.SCRIPTS / '626-arena-2026-11-04-baseline-disk-recovery.sql'
scalar = prior.scalar


def complete(p, lease, *, failed=False):
    if failed:
        return p.store.complete_attempt(run_id=lease['run_id'],lease_token_hash=hash_lease_token('source621'),
            result={'terminal_status':'provider_error'},terminal_cause='provider_error',output_ref=None)
    score=lease['kind']=='score'
    output={'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':lease['scored_run_id'],
        'breakdowns':[],'company_judgments':[]} if score else {'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':[]}
    payload=json.dumps(output).encode();ref=lease['run_id']+'.json';p.objects[ref]=payload
    return p.store.complete_attempt(run_id=lease['run_id'],lease_token_hash=hash_lease_token('source621'),
        result={'terminal_status':'accepted',**({'company_judgment_consumed_indexes':[]} if score else {})},
        terminal_cause='accepted',output_ref=ref,output_hash=contracts.hash_bytes(payload),
        **({'company_judgment_evidence':[]} if score else {}),completion_request_hash=contracts.document_hash({'run':lease['run_id']}))



def provider_bill(p, lease, marker):
    token=hash_lease_token('source621');identity='sha256:'+marker*64
    funding=p.store.get_run(lease['run_id'])['champion_funding_sources']['openrouter']
    def rpc(sql,args):
        with p.conn.cursor() as cur:
            cur.execute('SET ROLE lab_arena_service');cur.execute('SELECT public.'+sql,args)
            result=cur.fetchone()[0];cur.execute('RESET ROLE')
        p.conn.commit();return result
    reserve=lambda call:rpc("lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter',%s,100,%s::jsonb,1500)",(lease['run_id'],token,call,funding,json.dumps({'action_sequence':1,'request_hash':'sha256:'+'d'*64})))
    assert reserve(identity)['status']=='reserved'
    assert rpc('lab_arena_mark_dispatched(%s,%s,%s)',(lease['run_id'],token,identity))['status']=='dispatched'
    assert rpc('lab_arena_settle_call(%s,%s,%s,6000000,%s::jsonb,1500)',(lease['run_id'],token,identity,json.dumps({'status':200,'headers':{},'body_b64':base64.b64encode(b'{}').decode(),'call_succeeded':True})))['status']=='settled'
    if lease['attempt']==5:
        assert reserve('sha256:'+'c'*64)['reason']=='money_cap'



def assert_promotion_before_control(p, sql):
    """A627-style shared promotion owner must not deadlock with626."""
    ready=queue.Queue()
    def replay():
        with p.pg.connect(**p.dsn) as connection:
            ready.put(connection.get_backend_pid())
            with connection.cursor() as cursor:cursor.execute(sql)
            connection.commit()
    with p.pg.connect(**p.dsn) as owner:
        with owner.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock_shared(hashtextextended('lab_arena.baseline_promotion',0))")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(replay);pid=ready.get(timeout=3)
            try:
                deadline=time.monotonic()+3
                while True:
                    with p.conn.cursor() as cursor:
                        waiting=scalar(cursor,"SELECT count(*) FROM pg_locks WHERE pid=%s AND locktype='advisory' AND NOT granted",(pid,))
                    if waiting:break
                    assert time.monotonic()<deadline,'626 did not reach the promotion lock'
                    time.sleep(.01)
                # Before the fix,626 held claim-control while waiting for
                # promotion, which inverts627's shared-promotion→control order.
                with p.pg.connect(**p.dsn) as probe:
                    with probe.cursor() as cursor:
                        assert scalar(cursor,"SELECT pg_try_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0))") is True
                    probe.rollback()
            finally:owner.rollback()
            future.result(timeout=5)


def render(p):
    with p.conn.cursor() as cur:
        row=scalar(cur,"SELECT to_jsonb(r)-'updated_at' FROM public.lab_arena_rounds r WHERE round_id=%s",(ROUND,))
        hashes=scalar(cur,"SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) FROM public.lab_arena_runs a WHERE round_id=%s",(ROUND,))
        manifest={'round_hash':prior.hash_doc(cur,row),'run_hashes':hashes,'positions':[1,3,8],'future_hash':scalar(cur,"SELECT encode(extensions.digest((to_jsonb(r)-'updated_at')::TEXT,'sha256'),'hex') FROM public.lab_arena_rounds r WHERE round_id='arena-2026-11-05'")}
    text=re.sub(r'\$manifest626\$.*?\$manifest626\$',lambda _: '$manifest626$'+json.dumps(manifest,separators=(',',':'))+'$manifest626$',SQL.read_text(),flags=re.S)
    return text.replace('2026-11-05T01:00:00Z','2099-11-05T01:00:00Z').replace('2026-11-05T01:55:00Z','2099-11-05T01:55:00Z').replace('2026-11-05T02:00:00Z','2099-11-05T02:00:00Z')


@pytest.mark.parametrize('sixth_outcome',['accepted','judge_error'])
def test_exact_disk_recovery_native_lifecycle_preservation_and_replay(prepared,sixth_outcome):
    p=prepared;prior.apply(p)
    with p.conn.cursor() as cur:cur.execute((prior.SCRIPTS/'625-lab-arena-verified-fit-checkpoint.sql').read_text())
    p.conn.commit();prior.age_pending(p)
    for _ in range(10):
        lease=prior.claim(p);assert lease['kind']=='execute' and lease['attempt']==3
        assert complete(p,lease,failed=True)['status']=='failed'
    assert p.store.recover_baseline(ROUND)['queued']==10
    prior.age_pending(p)
    for _ in range(10):
        lease=prior.claim(p);assert lease['kind']=='execute' and lease['attempt']==4
        if lease['icp_position']==1:provider_bill(p,lease,'a')
        assert complete(p,lease)['status']=='accepted'
    p.service._recover_baseline(p.store.get_round(ROUND));prior.age_pending(p)
    for _ in range(10):
        lease=prior.claim(p);assert lease['kind']=='score'
        assert complete(p,lease)['status']=='accepted'
    assert p.service.close_scoring(ROUND,2)['status']=='closed'
    assert p.service.score_stage(ROUND,2)['status']=='ok'
    p.service._record_recovered_judgment_scores(p.store.get_round(ROUND))
    with p.conn.cursor() as cur:
        # Trusted runtime events are fixture facts; all repair behavior is native.
        for position in (1,3,8):
            for kind,content in [('runtime.stderr',{'text':'TYCHE runtime blocked: [Errno 28] No space left on device','sequence':0}),
                ('runtime.finished',{'exit_code':0,'resource_summary':{'validator_source_commit':'d2f2f757706df502070af7d6762ddea47cfda5a8'}})]:
                cur.execute("INSERT INTO public.lab_arena_trajectory_events(run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content) SELECT run_id,gen_random_uuid(),round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_position::text,stage,icp_position,attempt,kind,'baseline',%s,now(),%s::jsonb FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND kind='execute' AND icp_position=%s AND attempt=4",(kind,json.dumps(content),ROUND,BASELINE,position))
        # Exact unfinished ICP6 fixture: five retained failed judgments of the
        # clean execution. Their accepted/failed history is not repair behavior.
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',result_doc='{\"terminal_status\":\"judge_error\"}',output_ref=NULL WHERE round_id=%s AND submission_id=%s AND kind='score' AND icp_position=6 AND attempt=2",(ROUND,BASELINE))
        for attempt in (3,4,5):
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,stage_generation,kind,scored_run_id,terminal_cause,result_doc,company_judgment_refs,scoring_generation_floor) SELECT assignment_id||':'||%s,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,%s,'failed',stage_generation,kind,scored_run_id,'judge_error','{\"terminal_status\":\"judge_error\"}',company_judgment_refs,0 FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND kind='score' AND icp_position=6 AND attempt=2",(attempt,attempt,ROUND,BASELINE))
        cur.execute("UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE round_id=%s AND submission_id=%s AND kind='execute' AND icp_position=6 AND attempt=4",(ROUND,BASELINE))
        if sixth_outcome=='judge_error':
            # Preserve the real incident variant: the next day already has an
            # active leased baseline. Recovery must not rewind or replace it.
            future='arena-2026-11-05';future_baseline='baseline-2026-11-05'
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage1',status_generation=2,stage_generation=1,champion_funding_frozen=true,configuration_doc=jsonb_set(configuration_doc,'{baseline_source_url}',to_jsonb(%s::text)) WHERE round_id=%s",('https://github.com/leadpoet/leadpoet-sales-agent/archive/24d9201c3c5fdc86cb1676628774223205c87166.tar.gz',future))
            cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,is_king,status) SELECT %s,%s,miner_hotkey,true,'frozen' FROM public.lab_arena_submissions WHERE submission_id=%s",(future_baseline,future,BASELINE))
            cur.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(a)||jsonb_build_object('run_id',%s,'assignment_id',%s,'round_id',%s,'submission_id',%s,'attempt',1,'status','leased','stage_generation',1,'lease_expires_at',now()+interval '1 hour','result_doc',NULL,'terminal_cause',NULL,'output_ref',NULL,'per_icp_score',NULL,'qualification_doc',NULL))).* FROM public.lab_arena_runs a WHERE run_id=%s",(future+':'+future_baseline+':1:0:1',future+':'+future_baseline+':1:0',future,future_baseline,f'{ROUND}:{BASELINE}:1:0:4'))
        cur.execute('SET LOCAL session_replication_role=origin')
        future_before=scalar(cur,"SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id='arena-2026-11-05'")
        future_runs_before=scalar(cur,"SELECT coalesce(jsonb_agg(to_jsonb(a) ORDER BY run_id),'[]') FROM public.lab_arena_runs a WHERE round_id='arena-2026-11-05'")
        before=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))
        financial=scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
    p.conn.commit();sql=render(p)
    # A wrong manifest must roll back helper creation and all writes.
    wrong=sql.replace(json.loads(re.search(r'\$manifest626\$(.*?)\$manifest626\$',sql,re.S)[1])['round_hash'],'0'*64)
    with pytest.raises(p.pg.Error):
        with p.conn.cursor() as cur:cur.execute(wrong)
    p.conn.rollback()
    with p.conn.cursor() as cur:
        assert scalar(cur,"SELECT to_regprocedure('public.lab_arena__source_correction626_v1(public.lab_arena_runs)')") is None
        assert scalar(cur,"SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE request_id='nov04-baseline-disk626-v1'")==0
        cur.execute(sql)
    p.conn.commit()
    row=p.store.get_round(ROUND)
    assert row['status']=='stage2_scoring'
    assert row['configuration_doc']['schedule']['publication_deadline']=='2099-11-05T02:00:00Z'
    with p.conn.cursor() as cur:
        prior.assert_current_table_checks(cur)
        after=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))
        receipt=scalar(cur,"SELECT recovery_doc FROM public.lab_arena_scoring_supersessions WHERE request_id='nov04-baseline-disk626-v1'")
        assert len(receipt['superseded_runs'])==6 and len(after)==len(before)+4
        assert scalar(cur,"SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id='arena-2026-11-05'")==future_before
        assert scalar(cur,"SELECT coalesce(jsonb_agg(to_jsonb(a) ORDER BY run_id),'[]') FROM public.lab_arena_runs a WHERE round_id='arena-2026-11-05'")==future_runs_before
        for runid,original in before.items():
            omitted={'status','terminal_cause','terminal_doc','updated_at'} if runid in receipt['superseded_runs'] else set()
            assert {k:v for k,v in original.items()if k not in omitted}=={k:v for k,v in after[runid].items()if k not in omitted}
        assert scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')==financial
        cur.execute(sql)
        assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND status=\'pending\'',(ROUND,))==4
        for pos in (1,3,8):
            assert scalar(cur,"SELECT public.lab_arena__execute_infrastructure_cost_exempt_v1(r) FROM public.lab_arena_runs r WHERE run_id=%s",(f'{ROUND}:{BASELINE}:1:{pos}:4',)) is True
        assert scalar(cur,"SELECT public.lab_arena__execute_infrastructure_cost_exempt_v1(r) FROM public.lab_arena_runs r WHERE round_id=%s AND submission_id=%s AND icp_position=0 AND kind='execute' AND status='accepted'",(ROUND,BASELINE)) is False
        cur.execute('SET ROLE lab_arena_service')
        result=scalar(cur,"SELECT public.lab_arena_freeze_champion_funding('arena-2026-11-05')")
        if sixth_outcome=='accepted':assert result=={'status':'promotion_pending'}
        cur.execute('RESET ROLE')
    p.conn.commit()
    if sixth_outcome=='accepted':assert_promotion_before_control(p,sql)
    # Replay rejects security drift and missing authority instead of repairing it.
    for mutation in [
        'ALTER FUNCTION public.lab_arena__source_correction626_v1(public.lab_arena_runs) STRICT',
        'GRANT EXECUTE ON FUNCTION public.lab_arena__score_attempt_count_v1(public.lab_arena_runs) TO anon',
        'DROP FUNCTION public.lab_arena__configuration626_v1(jsonb,jsonb)',
    ]:
        with pytest.raises(p.pg.Error):
            with p.conn.cursor() as cur:cur.execute(mutation);cur.execute(sql)
        p.conn.rollback()
    prior.age_pending(p)
    executions=[];sixth=None
    for _ in range(4):
        lease=prior.claim(p);assert lease.get('kind')in('execute','score'),lease
        if lease['kind']=='execute':
            assert lease['attempt']==5 and lease['icp_position']in(1,3,8)
            assert p.service._is_baseline_recovery(p.store.get_run(lease['run_id']),p.store.list_runs(ROUND),row)
            if lease['icp_position']==1:provider_bill(p,lease,'b')
            assert complete(p,lease)['status']=='accepted';executions.append(lease)
        else:
            assert lease['attempt']==6 and lease['icp_position']==6 and lease['scored_run_id'].endswith(':4')
            sixth=lease
            if sixth_outcome=='judge_error':
                assert p.store.complete_attempt(run_id=lease['run_id'],lease_token_hash=hash_lease_token('source621'),
                    result={'terminal_status':'judge_error'},terminal_cause='judge_error',output_ref=None)['status']=='failed'
            else:assert complete(p,lease)['status']=='accepted'
    assert len(executions)==3 and sixth is not None
    p.service._recover_baseline(p.store.get_round(ROUND));prior.age_pending(p)
    for _ in range(3):
        lease=prior.claim(p);assert lease.get('kind')=='score' and lease['attempt']==3 and lease['scored_run_id'].endswith(':5'),lease
        assert complete(p,lease)['status']=='accepted'
    if sixth_outcome=='judge_error':
        p.store.expire_leases(ROUND)
        assert not any(r['attempt']>6 for r in p.store.list_runs(ROUND)if r['assignment_id']==sixth['assignment_id'])
        assert prior.claim(p)['status']=='no_pending'
        assert p.service.publish(ROUND)['status']=='stale'
        return
    assert p.service.close_scoring(ROUND,2)['status']=='closed'
    assert p.service.score_stage(ROUND,2)['status']=='ok'
    p.service._record_recovered_judgment_scores(p.store.get_round(ROUND))
    # Native raw financial cost remains12USD; the proved aborted generation
    # contributes6USD to the existing infrastructure exclusion, not comparison.
    with p.conn.cursor() as cur:
        state=scalar(cur,'SELECT public.lab_arena__successful_icp_cost_state(%s,%s,1)',(ROUND,BASELINE))
        assert state['settled_microusd']==12000000
        assert state['competition_settled_microusd']==6000000
        assert state['excluded_infrastructure_settled_microusd']==6000000
        assert scalar(cur,"SELECT public.lab_arena__score_attempt_count_v1(a) FROM public.lab_arena_runs a WHERE run_id=%s",(f'{ROUND}:{BASELINE}:1:6:score:6',))==5
        seventh=p.store.get_run(f'{ROUND}:{BASELINE}:1:6:score:6')
        seventh.update(attempt=7,run_id=f'{ROUND}:{BASELINE}:1:6:score:7')
        assert scalar(cur,'SELECT public.lab_arena__score_attempt_count_v1(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb))',(json.dumps(seventh),))==7
    from lab_arena.training_export import _competition_cost_snapshot,_source_evidence
    run=p.store.get_run(f'{ROUND}:{BASELINE}:1:1:5');current=p.store.get_round(ROUND)
    participant=next(x for x in current['participants']if x['is_king'])
    proof=_source_evidence(p.service._objects,run,current,p.store.get_submission(BASELINE),participant,p.store.scoring_policy_history(ROUND))
    scored=verify.scored_row(BASELINE,1,run['run_id'],p.service.evaluation_icps(ROUND)[1],[],[],current['configuration_doc']['scorer_policy'])
    observed={'observed_at':p.service.now().isoformat(),'eligibility':p.store.icp_cost_eligibility(round_id=ROUND,submission_id=BASELINE,icp_position=1,qualified_company_count=0)}
    result=_competition_cost_snapshot(run=run,configuration=current['configuration_doc'],scored=scored,
        related=[x for x in p.store.list_runs(ROUND)if x['assignment_id']==run['assignment_id']or x.get('scored_run_id')==run['run_id']],
        reader=lambda **_:observed,source_correction=proof['source_correction'])
    assert result['status']=='available' and result['competition_sourcing_microusd']==6000000,result
    publication=p.service.publish(ROUND)
    assert publication['status']=='ok',publication
    assert p.store.get_round(ROUND)['reward_activated_at'] is None
    history=p.store.scoring_policy_history(ROUND)
    assert all(f'{ROUND}:{BASELINE}:1:{pos}:5'in history['source_corrections'][0]['run_ids']for pos in(1,3,8))
    assert all(x['generation_floor']==0 for x in history['revisions'])
