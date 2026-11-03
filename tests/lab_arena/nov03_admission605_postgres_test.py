"""Exact original recovery through real review RPCs and the frozen stage2 contract."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from psycopg2.extras import Json

from lab_arena import code_review, submission_similarity
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.refusal_review_fallback599_postgres_test import failure, begin, PRIMARY, ALTERNATE
from tests.lab_arena.research_review_result566_postgres_test import _document
from tests.lab_arena.test_lab_arena_code_review_postgres import _age_review_error
from tests.lab_arena.test_lab_arena_migration_postgres import (
    encrypted_runtime_credentials, hotkey, round_config, source_submission_doc,
)
from tests.postgres_migration_harness import SCRIPTS
from tests.lab_arena.research_control_code_review_test import SOURCE, _assessment
from tests.lab_arena.test_lab_arena_code_review_runtime import _archive, _review_body, ReviewTransport
from tests.lab_arena.refusal_review_fallback_test import reviewer_for

ROUND='arena-2026-11-03'
TARGETS=('sub-28594d902080fb5213f4344c09ffb531','sub-ba551886a9379aa4e00b6238567f211a')
HASHES=('b0b27246c1b953694b1559737f1e78a9dc31dc4f3d55ff7d64b89567a00a64ae',
        '7a6ff726c4644dd3c28d94e2293967898c4d7dc9d6d9a86d20369463dc5a3b74')
SOURCE_HASHES=('sha256:cd4c88c88f38d6bc4140b3f64ca088fad246663f63d96555909fe9f589c6a894',
               'sha256:fc270d967ac39c835d9ccb2e5d909289911c9b38e3b919cf68149e18f032d967')
M605=SCRIPTS/'605-arena-2026-11-03-admission-review-recovery.sql'
M606=SCRIPTS/'606-arena-2026-11-03-admit-reviewed-originals.sql'


@pytest.fixture(scope='module')
def database():
    db=current_service_database()
    try:
        psycopg,dsn=next(db)
        with psycopg.connect(**dsn) as conn,conn.cursor() as cur:
            for file in ('594-lab-arena-private-review-diagnostics.sql','599-lab-arena-refusal-review-fallback.sql'):
                cur.execute((SCRIPTS/file).read_text())
        with psycopg.connect(**dsn) as conn,conn.cursor() as cur:
            cur.execute("SELECT string_agg(pg_get_functiondef(oid),E';\\n') FROM pg_proc WHERE oid IN "
                "('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::regprocedure,"
                "'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)'::regprocedure,"
                "'public.lab_arena_submissions_frozen_v1()'::regprocedure,"
                "'public.lab_arena_close_stage(text,smallint)'::regprocedure)")
            originals=cur.fetchone()[0]
        yield psycopg,dsn,originals
    finally:db.close()


@pytest.fixture()
def context(database,request):
    psycopg,dsn,originals=database
    conn=psycopg.connect(**dsn);conn.autocommit=True
    store=ArenaStore(PsycopgTransport(lambda:psycopg.connect(**dsn)))
    with conn.cursor() as cur:
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger CASCADE')
        # Restore only this disposable database's exact pre-605 bodies.
        cur.execute(originals)
        cur.execute('DROP FUNCTION IF EXISTS public.lab_arena_nov03_review_recovery_authorized_v1(text)')
        cur.execute('DROP FUNCTION IF EXISTS public.lab_arena__nov03_review605_v1(public.lab_arena_rounds,public.lab_arena_submissions)')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase=''")
    now=datetime.now(timezone.utc)
    deadline=(now+timedelta(hours=4)).isoformat()
    config=round_config(ROUND,[hotkey('runner605')])
    config['schedule'].update(submission_open=(now-timedelta(hours=2)).isoformat(),
        submission_cutoff=(now+timedelta(minutes=30)).isoformat(),
        benchmark_deadline=(now+timedelta(hours=1)).isoformat(),
        stage_2_close=(now+timedelta(seconds=getattr(request,'param',10800))).isoformat(),final_scoring_close=deadline)
    assert store.create_round(ROUND,config)['status']=='created'
    payload=_archive(SOURCE)
    inspected=submission_similarity.inspect_archive(payload)
    for i,target in enumerate(TARGETS):
        miner=hotkey('original605'+str(i))
        assert store.register_submission(ROUND,target,miner,{**source_submission_doc(ROUND,target),'source_size_bytes':len(payload)})['status']=='registered'
        assert store.accept_submission_with_credentials(ROUND,target,miner,encrypted_runtime_credentials(target))['status']=='ok'
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_submissions SET source_archive_sha256=%s,source_normalized_sha256=%s WHERE submission_id=%s",(inspected.archive_sha256,inspected.normalized_sha256,target))
        token=new_lease_token();assert begin(store,target,miner,PRIMARY,token)['status']=='claimed'
        doc={**failure(PRIMARY,'code_review_provider_request_rejected',False),'provider_http_status':200}
        assert store.finish_submission_review(target,miner,token,'error',doc,1234+i)['status']=='error'
        assert store.update_submission(ROUND,target,'accepted','rejected',{'rejection_rule':'code_review_incomplete'})['status']=='ok'
    participants=[dict(submission_id='existing605-'+str(i),miner_hotkey=hotkey('existing605-'+str(i)),
                       source_ref='arena/'+ROUND+'/sources/existing605-'+str(i)+'.tar.gz',source_size_bytes=4096,is_king=i==0)
                  for i in range(9)]
    with conn.cursor() as cur:
        cur.execute("SET session_replication_role='replica'")
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2',stage_generation=4,status_generation=5,participants=%s,"
                    "configuration_doc=configuration_doc||%s WHERE round_id=%s",
                    (Json(participants),Json({'execution_sequence_policy':'baseline_scored_first_v1',
                        'reasoning_capture_policy':'full_generated_thinking_v1'}),ROUND))
        cur.execute("SET session_replication_role='origin'")
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE,pause_reason='nov03_improved_verifier_required603'")
        cur.execute("INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,"
          "backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc) SELECT round_id,1,'nov03-forward-only-verifier604-v1',"
          "'sha256:'||repeat('a',64),'test605',txid_current(),pg_backend_pid(),to_jsonb(r),to_jsonb(r),'{}',"
          "'{\"operation\":\"nov03_forward_only_verifier_v1\"}' FROM public.lab_arena_rounds r WHERE round_id=%s",(ROUND,))
        migration=M605.read_text().replace('2026-11-03T20:30:02Z',deadline)
        for original_hash in SOURCE_HASHES:migration=migration.replace(original_hash,inspected.archive_sha256)
        for target,oldhash in zip(TARGETS,HASHES):
            cur.execute("SELECT encode(extensions.digest(to_jsonb(s)::text,'sha256'),'hex') FROM public.lab_arena_submissions s WHERE submission_id=%s",(target,))
            migration=migration.replace(oldhash,cur.fetchone()[0])
    yield store,conn,migration,M606.read_text().replace('2026-11-03T20:30:02Z',deadline),participants
    store._transport.close();conn.close()


def apply(conn,sql):
    try:
        with conn.cursor() as cur:cur.execute(sql)
    except Exception:
        with conn.cursor() as cur:cur.execute("ROLLBACK")
        raise


def snapshot(conn):
    with conn.cursor() as cur:
        result={}
        for table,key in (('lab_arena_rounds','round_id'),('lab_arena_submissions','submission_id'),
                          ('lab_arena_runs','run_id'),('lab_arena_ledger','entry_id'),
                          ('lab_arena_scoring_supersessions','generation_floor')):
            cur.execute(f"SELECT coalesce(jsonb_agg(to_jsonb(t) ORDER BY {key}),'[]'::jsonb) FROM public.{table} t")
            result[table]=cur.fetchone()[0]
        return result


def finish(store,conn,target,status='passed',model=PRIMARY,code='code_review_provider_refused'):
    row=store.get_submission(target);miner=row['miner_hotkey'];_age_review_error(conn,target)
    token=new_lease_token();assert begin(store,target,miner,model,token)['status']=='claimed'
    if status=='error':doc=failure(model,code,True)
    else:
        doc={**_document(status=status),'model':model,'file_count':2,'source_bytes':1000}
        doc['research_control']['source_archive_sha256']=row['source_archive_sha256']
        if status=='rejected':doc['categories']=['malicious_behavior']
    result=store.finish_submission_review(target,miner,token,status,doc,100)
    assert result['status']==status,result
    return token,doc


def test_exact_recovery_pass_reject_append_and_reapply_preserve_old_state(context):
    store,conn,m605,m606,parts=context
    before=snapshot(conn)
    with pytest.raises(ArenaStoreError,match='submission_missing'):
        begin(store,TARGETS[0],store.get_submission(TARGETS[0])['miner_hotkey'])
    apply(conn,m605)
    after=snapshot(conn)
    assert after['lab_arena_ledger']==before['lab_arena_ledger']
    assert after['lab_arena_rounds']==before['lab_arena_rounds']
    for old,new in zip(before['lab_arena_submissions'],after['lab_arena_submissions']):
        assert new==old|{'status':'accepted','updated_at':new['updated_at']}
    apply(conn,m605);assert snapshot(conn)==after
    finish(store,conn,TARGETS[0]);finish(store,conn,TARGETS[1],'rejected')
    reviewed=snapshot(conn)
    apply(conn,m606)
    final=snapshot(conn)
    assert final['lab_arena_rounds'][0]['participants']==parts+[
        {k:store.get_submission(TARGETS[0])[k] for k in ('submission_id','miner_hotkey','source_ref','source_size_bytes')}|{'is_king':False}]
    assert final['lab_arena_rounds'][0]['configuration_doc']==reviewed['lab_arena_rounds'][0]['configuration_doc']
    assert final['lab_arena_ledger']==reviewed['lab_arena_ledger']
    runs=final['lab_arena_runs'];assert len(runs)==20
    assert {r['icp_position'] for r in runs}==set(range(20))
    assert {r['stage'] for r in runs}=={2}
    assert {r['stage_generation'] for r in runs}=={4}
    assert store.get_submission(TARGETS[0])['status']=='frozen'
    assert store.get_submission(TARGETS[1])['status']=='rejected'
    assert store.get_submission(TARGETS[1])['code_review_status']=='rejected'
    apply(conn,m606);apply(conn,m605);assert snapshot(conn)==final


def test_refusal_episode_keeps_one_primary_plus_two_alternate_claims(context):
    store,conn,m605,m606,_=context;apply(conn,m605)
    original=store.list_ledger(submission_id=TARGETS[0])
    finish(store,conn,TARGETS[0],'error')
    row=store.get_submission(TARGETS[0]);assert row['code_review_doc']['review_primary_attempts']==2
    assert begin(store,TARGETS[0],row['miner_hotkey'],PRIMARY)['status']=='route_required'
    finish(store,conn,TARGETS[0],'error',ALTERNATE,'code_review_transport_error')
    finish(store,conn,TARGETS[0],'error',ALTERNATE)
    row=store.get_submission(TARGETS[0]);assert row['code_review_doc']['retryable'] is False
    ledger=store.list_ledger(submission_id=TARGETS[0]);assert ledger[:3]==original
    assert len(ledger)==12
    assert begin(store,TARGETS[0],row['miner_hotkey'],PRIMARY)['status']=='exhausted'
    assert begin(store,TARGETS[0],row['miner_hotkey'],ALTERNATE)['status']=='exhausted'
    finish(store,conn,TARGETS[1],'rejected')
    apply(conn,m606);assert not snapshot(conn)['lab_arena_runs']
    assert snapshot(conn)['lab_arena_scoring_supersessions'][-1]['recovery_doc']['excluded'][0]['code_review_status']=='error'


def test_concurrent_claims_and_same_claim_replay_reserve_once(context):
    store,conn,m605,_,_=context;apply(conn,m605);target=TARGETS[0]
    row=store.get_submission(target);_age_review_error(conn,target)
    tokens=[new_lease_token(),new_lease_token()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda token:begin(store,target,row['miner_hotkey'],PRIMARY,token),tokens))
    assert sorted(r['status'] for r in results)==['busy','claimed']
    token=tokens[next(i for i,r in enumerate(results) if r['status']=='claimed')]
    before=snapshot(conn);assert begin(store,target,row['miner_hotkey'],PRIMARY,token)['idempotent'] is True
    assert snapshot(conn)==before


@pytest.mark.parametrize('damage',['source','closed','deadline','unlisted','guard'])
def test_gates_deny_without_new_spend(context,damage):
    store,conn,m605,_,_=context;apply(conn,m605);target=TARGETS[0]
    with conn.cursor() as cur:
        if damage=='source':
            cur.execute("SET session_replication_role='replica'")
            cur.execute("UPDATE public.lab_arena_submissions SET source_archive_sha256='sha256:'||repeat('f',64) WHERE submission_id=%s",(target,))
            cur.execute("SET session_replication_role='origin'")
        elif damage in ('closed','deadline'):
            cur.execute("SET session_replication_role='replica'")
            if damage=='closed':cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_closed' WHERE round_id=%s",(ROUND,))
            else:cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s",(ROUND,))
            cur.execute("SET session_replication_role='origin'")
        elif damage=='guard':cur.execute("UPDATE public.lab_arena_restart_claim_control SET guard_commitment='sha256:'||repeat('f',64),owner_commitment='sha256:'||repeat('e',64),guard_expires_at=clock_timestamp()+interval '1 hour',candidate_commit=repeat('a',40),restart_scope='all',restart_phase='draining'")
        else:
            cur.execute("SET session_replication_role='replica'")
            cur.execute("UPDATE public.lab_arena_submissions SET submission_id='unlisted605' WHERE submission_id=%s",(target,))
            cur.execute("SET session_replication_role='origin'");target='unlisted605'
    row=store.get_submission(target);_age_review_error(conn,target)
    before=snapshot(conn)
    assert begin(store,target,row['miner_hotkey'])['status'] in ('round_closed','deadline','paused')
    assert snapshot(conn)==before


def test_finalize_requires_real_pass_source_proof_and_stopped_reviews(context):
    store,conn,m605,m606,_=context;apply(conn,m605)
    with pytest.raises(Exception,match='original or current review differs'):apply(conn,m606)
    finish(store,conn,TARGETS[0]);finish(store,conn,TARGETS[1],'rejected')
    with conn.cursor() as cur:cur.execute("UPDATE public.lab_arena_submissions SET code_review_doc=jsonb_set(code_review_doc,'{research_control,source_archive_sha256}',to_jsonb(('sha256:'||repeat('f',64))::text)) WHERE submission_id=%s",(TARGETS[0],))
    before=snapshot(conn)
    with pytest.raises(Exception,match='passing source proof differs'):apply(conn,m606)
    assert snapshot(conn)==before


def test_runtime_full_source_reader_hydration_and_service_worker(context):
    from types import SimpleNamespace
    import json
    from lab_arena.service import ArenaService
    store,conn,m605,m606,_=context
    payload=_archive(SOURCE)
    prepared=code_review.prepare_request(payload,research_control=True)
    def response(parameters):
        body=_review_body(parameters)
        content=json.loads(body['choices'][0]['message']['content'])
        content['research_control']=_assessment(prepared)
        body['choices'][0]['message']['content']=json.dumps(content)
        return {'body':body}
    transport=ReviewTransport(response);reviewer,_=reviewer_for(payload,transport,store)
    # A restored HTTP200 row cannot review before the exact receipt and reader.
    assert not store.get_submission(TARGETS[0])['code_review_doc'].get('legacy_refusal_proven')
    apply(conn,m605)
    for target in TARGETS:
        _age_review_error(conn,target)
        assert store.get_submission(target)['code_review_doc']['legacy_refusal_proven'] is True
    service=ArenaService.__new__(ArenaService)
    service._store=store;service._config=SimpleNamespace(code_reviewer=reviewer)
    service.active_rounds=lambda:[{'round_id':ROUND}]
    service._round=lambda round_id:store.get_round(round_id)
    service.now=lambda:datetime.now(timezone.utc)
    assert service.review_pending_submissions()=={'reviewed':2}
    assert len(transport.sent)==2
    assert all(store.get_submission(target)['code_review_status']=='passed' for target in TARGETS)
    apply(conn,m606)
    assert len(snapshot(conn)['lab_arena_runs'])==40


def test_finalize_preserves_running_other_lease_without_repausing(context):
    store,conn,m605,m606,parts=context;apply(conn,m605)
    finish(store,conn,TARGETS[0]);finish(store,conn,TARGETS[1],'rejected')
    with conn.cursor() as cur:
        cur.execute("SET session_replication_role='replica'")
        cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,source_ref,source_size_bytes) VALUES(%s,%s,%s,'frozen',%s,4096)",
                    (parts[1]['submission_id'],ROUND,parts[1]['miner_hotkey'],parts[1]['source_ref']))
        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,lease_generation,stage_generation,lease_token_hash,lease_expires_at,scoring_generation_floor) VALUES('ongoing605','ongoing605',%s,%s,%s,2,0,1,'execute','leased',1,4,%s,clock_timestamp()+interval '1 hour',0)",
                    (ROUND,parts[1]['submission_id'],parts[1]['miner_hotkey'],'sha256:'+'a'*64))
        cur.execute("SET session_replication_role='origin'")
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason=''")
    before=snapshot(conn);old=next(r for r in before['lab_arena_runs'] if r['run_id']=='ongoing605')
    apply(conn,m606);after=snapshot(conn)
    assert next(r for r in after['lab_arena_runs'] if r['run_id']=='ongoing605')==old
    assert after['lab_arena_ledger']==before['lab_arena_ledger']
    assert after['lab_arena_rounds'][0]['participants'][:9]==parts
    assert len(after['lab_arena_runs'])==21


def test_exact_function_security_and_unknown_body_fail_closed(context):
    store,conn,m605,_,_=context;apply(conn,m605)
    signature='public.lab_arena_nov03_review_recovery_authorized_v1(text)'
    with conn.cursor() as cur:
        cur.execute("SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE'),has_function_privilege('service_role',%s,'EXECUTE'),has_function_privilege('anon',%s,'EXECUTE'),has_function_privilege('authenticated',%s,'EXECUTE')",(signature,)*4)
        assert cur.fetchone()==(True,False,False,False)
        cur.execute("SELECT pg_get_functiondef('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::regprocedure)")
        cur.execute(cur.fetchone()[0].replace('BEGIN\n','BEGIN\n -- unknown605body\n',1))
    before=snapshot(conn)
    with pytest.raises(Exception,match='preimage differs'):apply(conn,m605)
    assert snapshot(conn)==before


def close_service(store):
    from unittest.mock import Mock
    from lab_arena.service import ArenaService
    service=ArenaService.__new__(ArenaService)
    service._store=store
    service._round=lambda round_id:store.get_round(round_id)
    # Exercise the actual service close controls and real SQL transition. The
    # next scoring-plan phase is independent of this admission-window guard.
    service.commit_scoring_plan=Mock(return_value={'status':'scoring_plan_reached'})
    return service


def completed_existing_execution(conn,parts):
    with conn.cursor() as cur:
        cur.execute("SET session_replication_role='replica'")
        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,scoring_generation_floor,terminal_cause,output_ref,result_doc) VALUES('complete605','complete605',%s,%s,%s,2,0,1,'execute','accepted',4,0,'accepted','arena/complete605.json','{\"terminal_status\":\"accepted\"}')",
                    (ROUND,parts[1]['submission_id'],parts[1]['miner_hotkey']))
        cur.execute("SET session_replication_role='origin'")
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason=''")


def test_service_cannot_close_complete_stage_while_reviews_or_finalization_remain(context):
    store,conn,m605,m606,parts=context;apply(conn,m605)
    completed_existing_execution(conn,parts)
    service=close_service(store)
    assert service.stage_is_complete(ROUND,2) is True
    before=snapshot(conn)
    waiting=service.close_stage(ROUND,2)
    assert waiting['status']=='waiting'
    assert waiting['reason']=='admission_recovery_pending'
    service.commit_scoring_plan.assert_not_called()
    assert snapshot(conn)==before
    # A finished review still needs explicit passed-only owner finalization.
    finish(store,conn,TARGETS[0]);finish(store,conn,TARGETS[1],'rejected')
    before=snapshot(conn)
    assert service.close_stage(ROUND,2)['status']=='waiting'
    assert snapshot(conn)==before
    apply(conn,m606)
    assert service.stage_is_complete(ROUND,2) is False  # Newly admitted work runs.
    with conn.cursor() as cur:
        cur.execute("SET session_replication_role='replica'")
        cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='arena/new605.json',result_doc='{\"terminal_status\":\"accepted\"}' WHERE status='pending'")
        cur.execute("SET session_replication_role='origin'")
    assert service.stage_is_complete(ROUND,2) is True
    assert service.close_stage(ROUND,2)=={'status':'scoring_plan_reached'}
    assert store.get_round(ROUND)['status']=='stage2_closed'
    service.commit_scoring_plan.assert_called_once_with(ROUND,2)


@pytest.mark.parametrize('context',[2],indirect=True)
def test_recovery_close_guard_expires_at_unchanged_stage2_deadline(context):
    import time
    store,conn,m605,_,parts=context;apply(conn,m605)
    completed_existing_execution(conn,parts)
    service=close_service(store)
    assert service.close_stage(ROUND,2)['status']=='waiting'
    deadline=datetime.fromisoformat(store.get_round(ROUND)['configuration_doc']['schedule']['stage_2_close'])
    time.sleep(max(0,(deadline-datetime.now(timezone.utc)).total_seconds())+0.02)
    assert service.close_stage(ROUND,2)=={'status':'scoring_plan_reached'}
    assert store.get_round(ROUND)['status']=='stage2_closed'


@pytest.mark.parametrize('damage',['source','unrelated_round'])
def test_recovery_close_guard_does_not_cover_unbound_source_or_other_round(context,damage):
    store,conn,m605,_,parts=context;apply(conn,m605)
    completed_existing_execution(conn,parts)
    service=close_service(store);round_id=ROUND
    with conn.cursor() as cur:
        if damage=='source':
            cur.execute("SET session_replication_role='replica'")
            cur.execute("UPDATE public.lab_arena_submissions SET source_archive_sha256='sha256:'||repeat('f',64) WHERE submission_id=%s",(TARGETS[0],))
            cur.execute("SET session_replication_role='origin'")
        else:
            round_id='arena-2099-03-01-unrelated605'
            assert store.create_round(round_id,round_config(round_id,[hotkey('other-runner605')]))['status']=='created'
            cur.execute("SET session_replication_role='replica'")
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id=%s",(round_id,))
            cur.execute("SET session_replication_role='origin'")
    assert service.close_stage(round_id,2)=={'status':'scoring_plan_reached'}
    assert store.get_round(round_id)['status']=='stage2_closed'
