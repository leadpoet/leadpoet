"""Add the ninth exact grant on top of the completed native635/636 lifecycle."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import json,re

import pytest
from psycopg2.extras import Json

from lab_arena import code_review_policy,submission_similarity,scoring
from lab_arena.store import new_lease_token
from tests.lab_arena.nov06_late_review635_postgres_test import (
    database,context as original_context,apply,row_hash,settle,M636,ROUND,
)
from tests.lab_arena.refusal_review_fallback599_postgres_test import PRIMARY,ALTERNATE,begin
from tests.lab_arena.test_lab_arena_code_review_postgres import _age_review_error
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey,source_submission_doc,encrypted_runtime_credentials
from tests.lab_arena.research_review_result566_postgres_test import _document
from tests.lab_arena.atomic_review_restart_guard587_postgres_test import acquire
from tests.postgres_migration_harness import SCRIPTS

SID='sub-43a929eb0e2e487f68014f48e7d80e6b'
M637=SCRIPTS/'637-arena-2026-11-06-ninth-review-recovery.sql'
M638=SCRIPTS/'638-arena-2026-11-06-admit-ninth-reviewed-original.sql'


@pytest.fixture
def context(database):
    fixture=original_context.__wrapped__(database)
    store,c,oldsql,oldtargets,participants,payload=next(fixture)
    try:
        # Extend the existing fixture before635's grant; only fixture source
        # admission is retimed. Restore the exact committed round afterward.
        round_before=store.get_round(ROUND)
        with c.cursor() as q:
            q.execute("SELECT pg_get_functiondef('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::regprocedure)")
            fixed=q.fetchone()[0]
            p634=(SCRIPTS/'634-lab-arena-invalid-review-alternate.sql').read_text()
            q.execute(fixed.replace(p634.split('$new634$')[1],p634.split('$old634$')[1]))
            q.execute("SET session_replication_role='replica'")
            q.execute("UPDATE public.lab_arena_rounds SET status='open' WHERE round_id=%s",(ROUND,))
            q.execute("SET session_replication_role='origin'")
        miner=hotkey('nov06-ninth637')
        assert store.register_submission(ROUND,SID,miner,{**source_submission_doc(ROUND,SID),'source_size_bytes':len(payload)})['status']=='registered'
        assert store.accept_submission_with_credentials(ROUND,SID,miner,encrypted_runtime_credentials(SID))['status']=='ok'
        source=submission_similarity.inspect_archive(payload)
        with c.cursor() as q:q.execute("UPDATE public.lab_arena_submissions SET source_archive_sha256=%s,source_normalized_sha256=%s WHERE submission_id=%s",(source.archive_sha256,source.normalized_sha256,SID))
        for attempt in range(1,4):
            if attempt>1:_age_review_error(c,SID)
            token=new_lease_token();assert begin(store,SID,miner,PRIMARY,token)['status']=='claimed'
            doc={'model':PRIMARY,'file_count':2,'source_bytes':1000,'error_code':'review_response_invalid','error_reason':'research_evidence_mismatch','retryable':True}
            assert store.finish_submission_review(SID,miner,token,'error',doc,1234)['status']=='error'
        _age_review_error(c,SID)
        assert store.update_submission(ROUND,SID,'accepted','rejected',{'rejection_rule':'code_review_incomplete'})['status']=='ok'
        # Production has three other unselected rows beyond18+8; preserve them too.
        for i in range(3):
            sid='unrelated-ninth637-'+str(i);mk=hotkey(sid)
            assert store.register_submission(ROUND,sid,mk,source_submission_doc(ROUND,sid))['status']=='registered'
            assert store.accept_submission_with_credentials(ROUND,sid,mk,encrypted_runtime_credentials(sid))['status']=='ok'
            assert store.update_submission(ROUND,sid,'accepted','rejected',{'rejection_rule':'code_review_incomplete'})['status']=='ok'
        with c.cursor() as q:
            q.execute(fixed)
            q.execute("SET session_replication_role='replica'")
            q.execute("UPDATE public.lab_arena_rounds SET status='committed',updated_at=%s WHERE round_id=%s",(round_before['updated_at'],ROUND))
            q.execute("SET session_replication_role='origin'")
        apply(c,oldsql)
        for x in oldtargets:settle(store,x['submission_id'],status='rejected',manifest=x['manifest_sha256'])
        apply(c,M636.read_text())
        with c.cursor() as q:q.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE WHERE singleton')
        baseline=next(p for p in participants if p['is_king'])
        assert store.open_stage(ROUND,1,[{'submission_id':baseline['submission_id'],'miner_hotkey':baseline['miner_hotkey']}],list(range(20)))['status']=='ok'
        store._ninth_preserved_runs=store.list_runs(ROUND)
        with c.cursor() as q:
            q.execute('SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton')
            store._ninth_preserved_control=q.fetchone()[0]
        sql=M637.read_text();target_text=sql.split('$targets635$')[1];target=json.loads(target_text)[0]
        with c.cursor() as q:
            target['row_sha256']=row_hash(q,'lab_arena_submissions','submission_id',SID)
            target['source_sha256']=source.archive_sha256
            target['manifest_sha256']=oldtargets[0]['manifest_sha256']
            q.execute("SELECT jsonb_agg(entry_id ORDER BY entry_id) FROM public.lab_arena_ledger WHERE submission_id=%s AND operation_id='openrouter.code_review'",(SID,));target['ledger_ids']=q.fetchone()[0]
            sql=sql.replace(target_text,json.dumps([target],separators=(',',':'),sort_keys=True))
            q.execute("SELECT submission_id,encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex') FROM public.lab_arena_submissions s WHERE round_id=%s AND submission_id<>%s",(ROUND,SID))
            preserved=dict(q.fetchall());assert len(preserved)==29
            sql=sql.replace(sql.split('$preserved635$')[1],json.dumps(preserved,separators=(',',':'),sort_keys=True))
            q.execute("SELECT request_id,request_hash FROM public.lab_arena_scoring_supersessions WHERE round_id=%s",(ROUND,))
            previous=dict(q.fetchall());assert len(previous)==2
            sql=sql.replace(sql.split('$previous637$')[1],json.dumps(previous,separators=(',',':'),sort_keys=True))
            q.execute("SELECT jsonb_agg(to_jsonb(h) ORDER BY generation_floor) FROM public.lab_arena_scoring_supersessions h WHERE round_id=%s",(ROUND,));old_receipts=q.fetchone()[0]
        old_rows={sid:store.get_submission(sid) for sid in preserved}
        old_ledgers={x['submission_id']:store.list_ledger(submission_id=x['submission_id']) for x in oldtargets}
        yield store,c,sql,target,oldtargets,participants,old_rows,old_ledgers,old_receipts
    finally:fixture.close()


def finish_ninth(store,target,status='error',token=None,cost=4321):
    row=store.get_submission(SID);token=token or new_lease_token()
    if row['code_review_status']!='reviewing':assert begin(store,SID,row['miner_hotkey'],ALTERNATE,token)['status']=='claimed'
    if status=='error':doc={'model':ALTERNATE,'file_count':2,'source_bytes':1000,'error_code':'review_response_invalid','error_reason':'research_evidence_mismatch','retryable':True}
    else:
        doc={**_document(status=status),'model':ALTERNATE,'file_count':2,'source_bytes':1000}
        doc['research_control'].update(source_archive_sha256=row['source_archive_sha256'],source_manifest_sha256=target['manifest_sha256'])
        if status=='rejected':doc['categories']=['malicious_behavior']
    return store.finish_submission_review(SID,row['miner_hotkey'],token,status,doc,cost)


def assert_preserved(context):
    store,c,_,_,_,participants,old_rows,old_ledgers,old_receipts=context
    assert all(store.get_submission(sid)==row for sid,row in old_rows.items())
    assert all(store.list_ledger(submission_id=sid)==ledger for sid,ledger in old_ledgers.items())
    with c.cursor() as q:
        q.execute("SELECT jsonb_agg(to_jsonb(h) ORDER BY generation_floor) FROM public.lab_arena_scoring_supersessions h WHERE round_id=%s AND request_id IN('nov06-late-review635-v1','nov06-admit-late-reviewed636-v1')",(ROUND,))
        assert q.fetchone()[0]==old_receipts
        q.execute('SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton')
        assert q.fetchone()[0]==store._ninth_preserved_control
    assert store.list_runs(ROUND)==store._ninth_preserved_runs
    assert store.get_round(ROUND)['participants'][:18]==participants


@pytest.mark.parametrize('status',['passed','rejected','error'])
def test_only_ninth_has_one_concurrent_fourth_and_strict_finalization(context,status):
    store,c,sql,target,oldtargets,_,_,_,_=context
    miner=store.get_submission(SID)['miner_hotkey']
    assert not code_review_policy.review_retry_available(store.get_submission(SID))
    with pytest.raises(Exception,match='submission_missing'):begin(store,SID,miner,ALTERNATE)
    apply(c,sql);apply(c,sql)
    assert code_review_policy.review_retry_available(store.get_submission(SID))
    assert code_review_policy.review_model_for(store.get_submission(SID))==ALTERNATE
    assert begin(store,SID,miner,PRIMARY)=={'status':'route_required','review_model':ALTERNATE}
    for x in oldtargets:
        row=store.get_submission(x['submission_id']);assert not code_review_policy.review_retry_available(row)
        with pytest.raises(Exception,match='submission_missing'):begin(store,x['submission_id'],row['miner_hotkey'],ALTERNATE)
    tokens=[new_lease_token(),new_lease_token()]
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda t:begin(store,SID,miner,ALTERNATE,t),tokens))
    assert sorted(r['status'] for r in results)==['busy','claimed']
    token=tokens[next(i for i,r in enumerate(results) if r['status']=='claimed')]
    assert begin(store,SID,miner,ALTERNATE,token)['idempotent'] is True
    assert finish_ninth(store,target,status,token)['status']==status
    if status=='error':
        _age_review_error(c,SID);assert begin(store,SID,miner,ALTERNATE)['status']=='exhausted'
    apply(c,sql)
    apply(c,M638.read_text());apply(c,M638.read_text())
    row=store.get_submission(SID)
    assert row['status']==('frozen' if status=='passed' else 'rejected')
    assert row['code_review_attempts']==4
    assert sum(r['entry_kind']=='reservation' for r in store.list_ledger(submission_id=SID))==4
    assert len(store.get_round(ROUND)['participants'])==(19 if status=='passed' else 18)
    assert [r['generation_floor'] for r in store.scoring_policy_history(ROUND)['revisions']]==[0,0,0,0]
    assert_preserved(context)


def test_late_cost_persists_without_admission_or_new_window(context):
    store,c,sql,target,_,_,_,_,_=context
    apply(c,sql);row=store.get_submission(SID);token=new_lease_token()
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE,token)['status']=='claimed'
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        q.execute("UPDATE public.lab_arena_scoring_supersessions SET created_at=clock_timestamp()-interval '31 minutes' WHERE request_id='nov06-ninth-review637-v1'")
        q.execute("UPDATE public.lab_arena_ledger SET created_at=clock_timestamp()-interval '30 minutes' WHERE submission_id=%s AND call_identity=(SELECT call_identity FROM public.lab_arena_ledger WHERE submission_id=%s AND entry_kind='reservation' AND entry_doc->'attempt'='4'::JSONB)",(SID,SID))
        q.execute("SET session_replication_role='origin'")
    assert finish_ninth(store,target,'passed',token)['status']=='deadline'
    row=store.get_submission(SID);assert row['code_review_doc']['cost_microusd']==4321 and row['code_review_doc']['cost_status']=='settled'
    apply(c,sql)
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status']=='deadline'
    apply(c,M638.read_text())
    assert store.get_submission(SID)['status']=='rejected'
    assert_preserved(context)


def test_expired_fourth_unknown_bill_is_excluded_without_fifth(context):
    store,c,sql,_,_,_,_,_,_=context
    apply(c,sql);row=store.get_submission(SID)
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status']=='claimed'
    with c.cursor() as q:q.execute("UPDATE public.lab_arena_submissions SET code_review_started_at=clock_timestamp()-interval '20 minutes',code_review_expires_at=clock_timestamp()-interval '1 second' WHERE submission_id=%s",(SID,))
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status']=='recovered'
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status']=='exhausted'
    apply(c,M638.read_text());assert store.get_submission(SID)['status']=='rejected'
    assert store.get_submission(SID)['code_review_doc']['cost_status']=='uncertain'
    assert_preserved(context)


def test_exact_request_hint_cannot_authorize_other_identity_or_prefix():
    now=datetime.now(timezone.utc)
    row={'submission_id':SID,'round_id':ROUND,'status':'accepted','code_review_status':'error','code_review_attempts':3,'code_review_doc':{
        'error_code':'review_response_invalid','review_invalid_attempts':3,'nov06_late_review':{'request_id':'nov06-ninth-review637-v1','expires_at':(now+timedelta(minutes=30)).isoformat()}}}
    assert code_review_policy.nov06_late_review_available(row,now=now)
    for field,value in [('submission_id','unrelated'),('round_id','arena-2026-11-07')]:assert not code_review_policy.nov06_late_review_available({**row,field:value},now=now)
    row['code_review_doc']['nov06_late_review']['request_id']+='-extra'
    assert not code_review_policy.nov06_late_review_available(row,now=now)


def expire_fixture_receipt(c):
    # Clock travel only this disposable new receipt; prior635/636 stay unchanged.
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        q.execute("UPDATE public.lab_arena_scoring_supersessions SET created_at=clock_timestamp()-interval '31 minutes',recovery_doc=jsonb_set(recovery_doc,'{expires_at}',to_jsonb((clock_timestamp()-interval '1 minute')::TEXT)) WHERE request_id='nov06-ninth-review637-v1'")
        q.execute("SET session_replication_role='origin'")


def test_unclaimed_ninth_expiry_excludes_without_fourth_or_extension(context):
    store,c,sql,_,_,_,_,_,_=context
    apply(c,sql);expire_fixture_receipt(c)
    row=store.get_submission(SID);before=store.list_ledger(submission_id=SID)
    hintrow={**row,'code_review_doc':{**row['code_review_doc'],'nov06_late_review':{
        'request_id':'nov06-ninth-review637-v1','expires_at':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}}}
    assert not code_review_policy.review_retry_available(hintrow)
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status']=='deadline'
    apply(c,sql)
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status']=='deadline'
    apply(c,M638.read_text())
    assert store.get_submission(SID)['code_review_attempts']==3
    assert store.get_submission(SID)['status']=='rejected'
    assert store.list_ledger(submission_id=SID)==before
    assert_preserved(context)


@pytest.mark.parametrize('fault',['hold','source','preserved','round','guard','stage1_plan'])
def test_new_receipt_rechecks_exact_state_before_spend_and_enrollment(context,fault):
    store,c,sql,_,oldtargets,participants,_,_,_=context
    apply(c,sql);row=store.get_submission(SID);before=store.list_ledger(submission_id=SID)
    with c.cursor() as q:
        if fault=='hold':q.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton')
        elif fault=='source':q.execute("UPDATE public.lab_arena_submissions SET source_size_bytes=source_size_bytes+1,submission_doc=jsonb_set(submission_doc,'{source_size_bytes}',to_jsonb(source_size_bytes+1)) WHERE submission_id=%s",(SID,))
        elif fault=='preserved':q.execute('UPDATE public.lab_arena_submissions SET updated_at=clock_timestamp() WHERE submission_id=%s',(oldtargets[0]['submission_id'],))
        elif fault=='round':
            q.execute("SET session_replication_role='replica'")
            q.execute("UPDATE public.lab_arena_rounds SET benchmark_ref=benchmark_ref||'.changed' WHERE round_id=%s",(ROUND,))
            q.execute("SET session_replication_role='origin'")
        elif fault=='stage1_plan':planning_fault(c,fault)
        else:acquire(c)
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status'] in {'exhausted','paused','round_closed','deadline'}
    assert store.list_ledger(submission_id=SID)==before
    with pytest.raises(Exception,match='differ'):apply(c,M638.read_text())


def test_ninth_cannot_borrow_old_receipt_window_or_reopen_other_rows(context):
    store,c,sql,_,oldtargets,_,_,_,_=context
    row=store.get_submission(SID);before=store.list_ledger(submission_id=SID)
    # Even a privileged fixture status forgery cannot borrow the original635
    # receipt: that exact receipt does not contain the ninth source.
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        q.execute("UPDATE public.lab_arena_submissions SET status='accepted' WHERE submission_id=%s",(SID,))
        q.execute("SET session_replication_role='origin'")
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status']=='round_closed'
    assert store.list_ledger(submission_id=SID)==before
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        q.execute("UPDATE public.lab_arena_submissions SET status='rejected',updated_at=%s WHERE submission_id=%s",(row['updated_at'],SID))
        q.execute("SET session_replication_role='origin'")
    apply(c,sql)
    with pytest.raises(Exception,match='cannot be reopened'):
        with c.cursor() as q:q.execute("UPDATE public.lab_arena_submissions SET status='accepted' WHERE submission_id=%s",(oldtargets[0]['submission_id'],))
    assert_preserved(context)


def test_ninth_wrong_full_source_proof_never_enrolls(context):
    store,c,sql,target,_,_,_,_,_=context
    apply(c,sql)
    bad={**target,'manifest_sha256':'sha256:'+'0'*64}
    assert finish_ninth(store,bad,'passed')['status']=='passed'
    with pytest.raises(Exception,match='passing full-source proof differs'):apply(c,M638.read_text())
    assert len(store.get_round(ROUND)['participants'])==18
    assert store.get_submission(SID)['status']=='accepted'
    assert_preserved(context)


def test_new_selector_preserves_inflight_finish_under_canonical_guard(context):
    store,c,sql,target,_,_,_,_,_=context
    apply(c,sql);row=store.get_submission(SID);token=new_lease_token()
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE,token)['status']=='claimed'
    acquire(c)
    assert finish_ninth(store,target,'passed',token)['status']=='passed'
    with pytest.raises(Exception,match='differ'):apply(c,M638.read_text())
    with c.cursor() as q:q.execute("UPDATE public.lab_arena_restart_claim_control SET guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'::JSONB WHERE singleton")
    with c.cursor() as q:
        q.execute('SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton')
        store._ninth_preserved_control=q.fetchone()[0]
    apply(c,M638.read_text())
    assert store.get_submission(SID)['status']=='frozen'
    assert_preserved(context)


def test_baseline_progress_survives_review_and_pass_enters_normal_miner_plan(context):
    store,c,sql,target,_,_,_,_,_=context
    with c.cursor() as q:
        q.execute("SELECT pg_get_functiondef('public.lab_arena_rounds_write_once_v1()'::regprocedure)")
        assert "h.old_round_doc=to_jsonb(OLD)-'updated_at'" in q.fetchone()[0]
    apply(c,sql)
    row=store.get_submission(SID);token=new_lease_token()
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE,token)['status']=='claimed'
    # Use the native phase transition: source and baseline rows may legitimately
    # advance during the review. The recovery must preserve their new state.
    assert store.close_stage(ROUND,1)['status']=='closed'
    store._ninth_preserved_runs=store.list_runs(ROUND)
    assert store.get_round(ROUND)['status']=='stage1_closed'
    current=store.get_round(ROUND)
    plan=scoring.build_scoring_plan(round_id=ROUND,stage=1,runs=store._ninth_preserved_runs,
        execution_sequence_policy='baseline_scored_first_v1',configuration=current['configuration_doc'])
    assert store.transition_round(ROUND,'stage1_closed','stage1_closed',{'stage1_scoring_plan_doc':plan})['status']=='ok'
    assert finish_ninth(store,target,'passed',token)['status']=='passed'
    apply(c,M638.read_text());assert_preserved(context)
    participants=store.get_round(ROUND)['participants']
    miners=[{'submission_id':p['submission_id'],'miner_hotkey':p['miner_hotkey']} for p in participants if not p['is_king']]
    # Complete only the disposable phase fixture; the tested stage2 planner is
    # the unmodified native path and must include all current frozen miners.
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        q.execute("UPDATE public.lab_arena_rounds SET status='stage1_scored',finalists=%s WHERE round_id=%s",(Json([p['submission_id'] for p in miners]),ROUND))
        q.execute("SET session_replication_role='origin'")
    with pytest.raises(Exception,match='stage_participants_invalid'):
        store.open_stage(ROUND,2,[p for p in miners if p['submission_id']!=SID],list(range(20)))
    assert store.list_runs(ROUND,stage=2)==[]
    assert store.open_stage(ROUND,2,miners,list(range(20)))['status']=='ok'
    assert len(store.list_runs(ROUND,stage=2,submission_id=SID))==20
    assert store.list_runs(ROUND,stage=1)==store._ninth_preserved_runs


def planning_fault(c, fault):
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        if fault=='phase':q.execute("UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id=%s",(ROUND,))
        elif fault=='plan':q.execute("UPDATE public.lab_arena_rounds SET stage2_scoring_plan_doc='{}' WHERE round_id=%s",(ROUND,))
        elif fault=='finalists':q.execute("UPDATE public.lab_arena_rounds SET finalists='[]' WHERE round_id=%s",(ROUND,))
        elif fault=='stage1_plan':q.execute("UPDATE public.lab_arena_rounds SET stage1_scoring_plan_doc=%s WHERE round_id=%s",(Json({'work_items':[{'submission_id':'preserved-late635-1'}],'zero_rows':[]}),ROUND))
        elif fault=='miner':q.execute("UPDATE public.lab_arena_runs SET submission_id='preserved-late635-1' WHERE run_id=(SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s LIMIT 1)",(ROUND,))
        else:q.execute("UPDATE public.lab_arena_runs SET stage=2 WHERE run_id=(SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s LIMIT 1)",(ROUND,))
        q.execute("SET session_replication_role='origin'")


@pytest.mark.parametrize('fault',['phase','plan','finalists','miner','stage2','stage1_plan'])
@pytest.mark.parametrize('at_admission',[False,True])
def test_existing_miner_planning_blocks_grant_or_admission(context,fault,at_admission):
    store,c,sql,target,_,participants,_,_,_=context
    if at_admission:
        apply(c,sql);assert finish_ninth(store,target,'passed')['status']=='passed'
    before=store.list_ledger(submission_id=SID)
    planning_fault(c,fault)
    with pytest.raises(Exception,match='differ'):apply(c,M638.read_text() if at_admission else sql)
    assert store.get_round(ROUND)['participants']==participants
    assert store.list_ledger(submission_id=SID)==before


@pytest.mark.parametrize('late_admission',[False,True])
def test_after_stage1_close_real_claim_settlement_and_bounded_admission(context,late_admission):
    store,c,sql,target,_,participants,_,_,_=context
    close=datetime.fromisoformat(store.get_round(ROUND)['configuration_doc']['schedule']['stage_1_close'].replace('Z','+00:00'))
    moment=close+timedelta(seconds=1)
    literal="TIMESTAMPTZ '%s'"%moment.isoformat()
    # Travel only this disposable DB's review clocks and ledger insert clock.
    # Keep the original frozen schedule and every old receipt/source byte.
    functions=[]
    with c.cursor() as q:
        for signature in ('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)',
                          'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)'):
            q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(signature,));definition=q.fetchone()[0];functions.append(definition)
        q.execute("SELECT pg_get_expr(d.adbin,d.adrelid) FROM pg_attrdef d JOIN pg_attribute a ON a.attrelid=d.adrelid AND a.attnum=d.adnum WHERE d.adrelid='public.lab_arena_ledger'::regclass AND a.attname='created_at'")
        default=q.fetchone()[0]
    try:
        apply(c,sql.replace('created TIMESTAMPTZ:=clock_timestamp()','created TIMESTAMPTZ:='+literal))
        with c.cursor() as q:
            for definition in functions:q.execute(definition.replace('pg_catalog.clock_timestamp()',literal))
            q.execute('ALTER TABLE public.lab_arena_ledger ALTER COLUMN created_at SET DEFAULT '+literal)
        row=store.get_submission(SID);token=new_lease_token()
        assert datetime.fromisoformat(row['code_review_doc']['nov06_late_review']['expires_at'])==moment+timedelta(minutes=30)
        assert begin(store,SID,row['miner_hotkey'],ALTERNATE,token)['status']=='claimed'
        assert finish_ninth(store,target,'passed',token)['status']=='passed'
        assert store.get_submission(SID)['code_review_doc']['cost_status']=='settled'
        assert store.get_submission(SID)['code_review_doc']['cost_microusd']==4321
        if late_admission:
            expired="TIMESTAMPTZ '%s'"%(moment+timedelta(minutes=30)).isoformat()
            with pytest.raises(Exception,match='passing full-source proof differs'):
                apply(c,M638.read_text().replace('clock_timestamp()',expired))
            assert store.get_round(ROUND)['participants']==participants
        else:
            apply(c,M638.read_text().replace('clock_timestamp()',literal))
            assert store.get_submission(SID)['status']=='frozen'
            assert len(store.get_round(ROUND)['participants'])==19
        assert_preserved(context)
    finally:
        with c.cursor() as q:
            for definition in functions:q.execute(definition)
            q.execute('ALTER TABLE public.lab_arena_ledger ALTER COLUMN created_at SET DEFAULT '+default)
