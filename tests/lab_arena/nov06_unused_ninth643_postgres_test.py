"""Exact unused637 closure after planning, with no new paid or admission authority."""
import json
import pytest
from psycopg2.extras import Json
from tests.lab_arena.nov06_ninth_review637_postgres_test import (
 database,context,apply,finish_ninth,SID,M638,ROUND,
)
from tests.lab_arena.refusal_review_fallback599_postgres_test import begin,ALTERNATE
from tests.postgres_migration_harness import SCRIPTS

SQL=SCRIPTS/'643-arena-2026-11-06-finalize-unspent-ninth-review.sql'


def prepared(ctx, *, advance=True, fourth=False):
    store,c,grant,target,*_=ctx
    apply(c,grant)
    if fourth:
        row=store.get_submission(SID)
        assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status']=='claimed'
    with c.cursor() as q:
        q.execute("SELECT request_hash,created_at FROM public.lab_arena_scoring_supersessions WHERE request_id='nov06-ninth-review637-v1'")
        request_hash,created=q.fetchone()
    text=SQL.read_text().replace('sha256:dda642385905dab62b935c84aa21ef99fb3c0ef1f2f16e058dc9cd0ebfa72cfe',request_hash).replace('2026-11-06T05:26:24.67524Z',created.isoformat())
    text=text.replace('sha256:5ae54960007f2cc87130634b62f5dff3cbe40d5978dd59640fce4574b9855301',target['source_sha256']).replace('sha256:30d0d5baddcde4edc7f11ad17a5724108dd9138bf3115a4cbf9eca0d9bee06fd',target['manifest_sha256'])
    if advance:
        participants=store.get_round(ROUND)['participants']
        miners=[dict(submission_id=p['submission_id'],miner_hotkey=p['miner_hotkey']) for p in participants if not p['is_king']]
        with c.cursor() as q:
            q.execute("SET session_replication_role='replica'")
            q.execute("UPDATE public.lab_arena_rounds SET status='stage1_scored',finalists=%s WHERE round_id=%s",(Json([p['submission_id'] for p in miners]),ROUND))
            q.execute("SET session_replication_role='origin'")
        assert store.open_stage(ROUND,2,miners,list(range(20)))['status']=='ok'
    return text


def snapshot(store,c):
    with c.cursor() as q:
        q.execute('SELECT to_jsonb(t) FROM public.lab_arena_restart_claim_control t WHERE singleton');control=q.fetchone()[0]
        q.execute('SELECT jsonb_agg(to_jsonb(t) ORDER BY generation_floor) FROM public.lab_arena_scoring_supersessions t WHERE round_id=%s',(ROUND,));receipts=q.fetchone()[0]
    return dict(round=store.get_round(ROUND),rows={s['submission_id']:s for status in ('accepted','frozen','rejected') for s in store.list_submissions(ROUND,status=status)},runs=store.list_runs(ROUND),ledger=store.list_ledger(),control=control,receipts=receipts)


def test_closure_before_after_exact_preservation_replay_and_no_paid_claim(context):
    store,c,*_=context
    sql=prepared(context)
    before=snapshot(store,c)
    row=store.get_submission(SID)
    assert begin(store,SID,row['miner_hotkey'],ALTERNATE)['status']=='round_closed'
    with pytest.raises(Exception,match='differ'):apply(c,M638.read_text())
    assert snapshot(store,c)==before
    apply(c,sql)
    after=snapshot(store,c)
    assert after['round']==before['round'] and after['runs']==before['runs']
    assert after['ledger']==before['ledger'] and after['control']==before['control']
    assert len(after['receipts'])==len(before['receipts'])+1
    assert after['receipts'][:-1]==before['receipts']
    for sid,original in before['rows'].items():
        if sid==SID:
            assert after['rows'][sid]['status']=='rejected'
            assert after['rows'][sid]['rejection_rule']=='code_review_incomplete'
            ignore={'status','rejection_rule','updated_at'}
            assert {k:v for k,v in after['rows'][sid].items() if k not in ignore}=={k:v for k,v in original.items() if k not in ignore}
        else:assert after['rows'][sid]==original
    receipt=after['receipts'][-1]
    assert receipt['recovery_doc']['admitted']==[]
    assert receipt['recovery_doc']['excluded']==[SID]
    proof=receipt['recovery_doc']['reviews'][SID]
    assert proof['outcome']=='preminer_gate_closed_without_fourth'
    assert proof['additional_reservations']==proof['additional_cost_microusd']==0
    assert [r['generation_floor'] for r in store.scoring_policy_history(ROUND)['revisions']]==[0,0,0,0]
    apply(c,sql);apply(c,M638.read_text())
    assert snapshot(store,c)==after
    with pytest.raises(Exception,match='submission_missing'):begin(store,SID,row['miner_hotkey'],ALTERNATE)
    assert store.list_ledger()==before['ledger']


@pytest.mark.parametrize('fault',['fourth','passed','source','not_advanced','other','guard','different_receipt'])
def test_closure_rejects_changed_authority_without_mutation(context,fault):
    store,c,_,target,*_=context
    sql=prepared(context,advance=fault!='not_advanced',fourth=fault=='fourth')
    with c.cursor() as q:
        if fault=='passed':
            q.execute("SET session_replication_role='replica'")
            q.execute("UPDATE public.lab_arena_submissions SET code_review_status='passed',code_review_doc=code_review_doc||'{\"verdict\":\"pass\",\"passed\":true}'::JSONB WHERE submission_id=%s",(SID,))
            q.execute("SET session_replication_role='origin'")
        elif fault=='source':q.execute("UPDATE public.lab_arena_submissions SET source_size_bytes=source_size_bytes+1,submission_doc=jsonb_set(submission_doc,'{source_size_bytes}',to_jsonb(source_size_bytes+1)) WHERE submission_id=%s",(SID,))
        elif fault=='other':q.execute("UPDATE public.lab_arena_submissions SET updated_at=clock_timestamp() WHERE submission_id='preserved-late635-1'")
        elif fault=='guard':
            from tests.lab_arena.atomic_review_restart_guard587_postgres_test import acquire
            acquire(c)
        elif fault=='different_receipt':
            q.execute("INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc) SELECT round_id,4,'nov06-admit-ninth-reviewed638-v1',%s,'wrong-actor',txid_current(),pg_backend_pid(),to_jsonb(r),to_jsonb(r),'{}','{}' FROM public.lab_arena_rounds r WHERE round_id=%s",('sha256:'+'1'*64,ROUND))
    before=snapshot(store,c)
    with pytest.raises(Exception,match='differ'):apply(c,sql)
    assert snapshot(store,c)==before
