"""Receipt-bound late review and enrollment on the native PostgreSQL paths.

Fixture substitutions are limited to native row hashes/ledger IDs and clocks.
No provider, production connection, original source, or scoring rule is changed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import re

import pytest
from psycopg2.extras import Json

from lab_arena import code_review, code_review_policy, submission_similarity
from lab_arena.store import ArenaStore, PsycopgTransport, new_lease_token
from tests.lab_arena.invalid_review_budget624_postgres_test import recovery_database as predecessor_database, finish
from tests.lab_arena.refusal_review_fallback599_postgres_test import PRIMARY, ALTERNATE, begin, verdict
from tests.lab_arena.test_lab_arena_code_review_postgres import _age_review_error
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, round_config, source_submission_doc, encrypted_runtime_credentials
from tests.lab_arena.test_lab_arena_code_review import _archive
from tests.lab_arena.research_control_code_review_test import SOURCE
from tests.lab_arena.research_review_result566_postgres_test import _document
from tests.postgres_migration_harness import SCRIPTS
from tests.lab_arena.atomic_review_restart_guard587_postgres_test import acquire

M635 = SCRIPTS/'635-arena-2026-11-06-late-review-recovery.sql'
M636 = SCRIPTS/'636-arena-2026-11-06-admit-late-reviewed-originals.sql'
ROUND = 'arena-2026-11-06'


def apply(connection, sql):
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
    except Exception:
        with connection.cursor() as cursor:
            cursor.execute('ROLLBACK')
        raise


def row_hash(cursor, table, key, value):
    cursor.execute(f"SELECT encode(extensions.digest(to_jsonb(t)::TEXT,'sha256'),'hex') FROM public.{table} t WHERE {key}=%s", (value,))
    return cursor.fetchone()[0]


@pytest.fixture(scope='module')
def database():
    f=predecessor_database.__wrapped__()
    try:
        pg,dsn=next(f)
        with pg.connect(**dsn) as c,c.cursor() as q:
            q.execute((SCRIPTS/'634-lab-arena-invalid-review-alternate.sql').read_text())
            source621=(SCRIPTS/'621-arena-2026-11-04-baseline-source-correction.sql').read_text()
            q.execute(source621[source621.index('CREATE OR REPLACE FUNCTION public.lab_arena__source_correction621_v1('):source621.index('CREATE TEMP TABLE arena621_final_functions')])
            q.execute((SCRIPTS/'630-arena-2026-11-04-baseline-donor-evidence.sql').read_text().split('$definition630_6$')[1])
            q.execute("SELECT pg_get_functiondef('public.lab_arena_submissions_frozen_v1()'::regprocedure)")
            frozen=q.fetchone()[0]
            q.execute(frozen.replace(source621.split('$old621_9$')[1],source621.split('$new621_9$')[1]))
            q.execute("SELECT string_agg(pg_get_functiondef(oid),E';\\n') FROM pg_proc WHERE oid IN "
                "('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::regprocedure,"
                "'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)'::regprocedure,"
                "'public.lab_arena_submissions_frozen_v1()'::regprocedure,"
                "'public.lab_arena_scoring_policy_history_v1(text)'::regprocedure)")
            originals=q.fetchone()[0]
        yield pg,dsn,originals
    finally:f.close()


@pytest.fixture
def context(database):
    pg,dsn,originals=database
    c=pg.connect(**dsn);c.autocommit=True
    store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
    with c.cursor() as q:
        q.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger CASCADE')
        q.execute(originals)
        q.execute('DROP FUNCTION IF EXISTS public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)')
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'::JSONB WHERE singleton")
        # Set up old three-primary failures against the true predecessor;
        # current634 would correctly route the third invalid claim to Sol.
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)', ('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)',))
        fixed=q.fetchone()[0]
        p634=(SCRIPTS/'634-lab-arena-invalid-review-alternate.sql').read_text()
        q.execute(fixed.replace(p634.split('$new634$')[1],p634.split('$old634$')[1]))
    config=round_config(ROUND,[hotkey('late635-runner')])
    now=datetime.now(timezone.utc)
    config['schedule'].update(submission_open=(now-timedelta(hours=2)).isoformat(),
        submission_cutoff=(now+timedelta(minutes=30)).isoformat(), benchmark_deadline=(now+timedelta(hours=1)).isoformat(),
        stage_1_close=(now+timedelta(hours=4)).isoformat())
    assert store.create_round(ROUND,config)['status']=='created'
    sql=M635.read_text();expected_text=sql.split('$targets635$')[1];expected=json.loads(expected_text)
    payload=_archive(SOURCE);inspected=submission_similarity.inspect_archive(payload)
    manifests={}
    for i,item in enumerate(expected):
        sid=item['submission_id'];miner=hotkey('late635-'+str(i))
        assert store.register_submission(ROUND,sid,miner,{**source_submission_doc(ROUND,sid),'source_size_bytes':len(payload)})['status']=='registered'
        assert store.accept_submission_with_credentials(ROUND,sid,miner,encrypted_runtime_credentials(sid))['status']=='ok'
        with c.cursor() as q:
            q.execute('UPDATE public.lab_arena_submissions SET source_archive_sha256=%s,source_normalized_sha256=%s WHERE submission_id=%s',
                (inspected.archive_sha256,inspected.normalized_sha256,sid))
        for attempt in range(1,4):
            if attempt>1:_age_review_error(c,sid)
            token=new_lease_token();assert begin(store,sid,miner,PRIMARY,token)['status']=='claimed'
            assert finish(store,sid,miner,token,error_reason='research_evidence')['status']=='error'
        _age_review_error(c,sid)
        assert store.update_submission(ROUND,sid,'accepted','rejected',{'rejection_rule':'code_review_incomplete'})['status']=='ok'
        with c.cursor() as q:
            item['row_sha256']=row_hash(q,'lab_arena_submissions','submission_id',sid)
            q.execute("SELECT jsonb_agg(entry_id ORDER BY entry_id) FROM public.lab_arena_ledger WHERE submission_id=%s AND operation_id='openrouter.code_review'",(sid,));item['ledger_ids']=q.fetchone()[0]
        item['source_sha256']=inspected.archive_sha256
        item['manifest_sha256']=code_review.prepare_request(payload,research_control=True).source_manifest_sha256
        manifests[sid]=item['manifest_sha256']
    participants=[]
    for i in range(18):
        sid='baseline-2026-11-06' if i==0 else 'preserved-late635-'+str(i)
        miner=hotkey('preserved-late635-'+str(i))
        assert store.register_submission(ROUND,sid,miner,source_submission_doc(ROUND,sid))['status']=='registered'
        assert store.accept_submission_with_credentials(ROUND,sid,miner,encrypted_runtime_credentials(sid))['status']=='ok'
        participants.append(dict(submission_id=sid,miner_hotkey=miner,source_ref=source_submission_doc(ROUND,sid)['source_ref'],source_size_bytes=4096,is_king=i==0))
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        q.execute("UPDATE public.lab_arena_submissions SET status='frozen',frozen_at=clock_timestamp(),is_king=(submission_id='baseline-2026-11-06') WHERE submission_id LIKE 'preserved-late635-%%' OR submission_id='baseline-2026-11-06'")
        q.execute("UPDATE public.lab_arena_rounds SET status='committed',status_generation=1,participants=%s,benchmark_ref=%s,"
            "configuration_doc=configuration_doc||%s WHERE round_id=%s",(Json(participants),'arena/'+ROUND+'/benchmark.json',Json({'reasoning_capture_policy':'full_generated_thinking_v1','execution_sequence_policy':'baseline_scored_first_v1'}),ROUND))
        q.execute("SET session_replication_role='origin'")
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE,pause_reason='await_reviewed_recovery633_before_publication' WHERE singleton")
        sql=sql.replace(re.search(r"to_jsonb\(r\)::TEXT,'sha256'\),'hex'\)<>'([0-9a-f]{64})'",sql).group(1),row_hash(q,'lab_arena_rounds','round_id',ROUND))
        preserved={x['submission_id']:row_hash(q,'lab_arena_submissions','submission_id',x['submission_id']) for x in participants}
        sql=sql.replace(sql.split('$preserved635$')[1],json.dumps(preserved,separators=(',',':'),sort_keys=True))
        q.execute(fixed)
    sql=sql.replace(expected_text,json.dumps(expected,separators=(',',':'),sort_keys=True))
    yield store,c,sql,expected,participants,payload
    store.close();c.close()


def settle(store,sid,*,status='error',manifest=None):
    row=store.get_submission(sid);token=new_lease_token()
    claim=begin(store,sid,row['miner_hotkey'],ALTERNATE,token)
    assert claim['status']=='claimed',claim
    if status=='error':doc={'model':ALTERNATE,'file_count':2,'source_bytes':1000,'error_code':'review_response_invalid','error_reason':'research_evidence','retryable':True}
    else:
        doc={**_document(status=status),'model':ALTERNATE,'file_count':2,'source_bytes':1000}
        doc['research_control'].update(source_archive_sha256=row['source_archive_sha256'],source_manifest_sha256=manifest)
        if status=='rejected':doc['categories']=['malicious_behavior']
    result=store.finish_submission_review(sid,row['miner_hotkey'],token,status,doc,4321)
    assert result['status']==status,result
    return token


def test_exact_window_claim_concurrency_terminal_cap_and_pass_enrollment(context):
    store,c,sql,expected,participants,_=context
    before={x['submission_id']:store.list_ledger(submission_id=x['submission_id']) for x in expected}
    apply(c,sql);apply(c,sql)
    assert [x['generation_floor'] for x in store.scoring_policy_history(ROUND)['revisions']]==[0]
    for x in expected:
        row=store.get_submission(x['submission_id'])
        assert row['status']=='accepted' and row['code_review_attempts']==3
        assert code_review_policy.review_retry_available(row)
        assert code_review_policy.review_model_for(row)==ALTERNATE
        assert store.list_ledger(submission_id=x['submission_id'])==before[x['submission_id']]
    sid=expected[0]['submission_id'];miner=store.get_submission(sid)['miner_hotkey']
    assert begin(store,sid,miner,PRIMARY)=={'status':'route_required','review_model':ALTERNATE}
    tokens=[new_lease_token(),new_lease_token()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda token:begin(store,sid,miner,ALTERNATE,token),tokens))
    assert sorted(x['status'] for x in results)==['busy','claimed']
    token=tokens[next(i for i,x in enumerate(results) if x['status']=='claimed')]
    assert begin(store,sid,miner,ALTERNATE,token)['idempotent'] is True
    assert finish(store,sid,miner,token,model=ALTERNATE)['status']=='error'
    _age_review_error(c,sid)
    assert begin(store,sid,miner,ALTERNATE)['status']=='exhausted'
    apply(c,sql)
    assert begin(store,sid,miner,ALTERNATE)['status']=='exhausted'
    with pytest.raises(Exception,match='unspent review window remains'):
        apply(c,M636.read_text())
    for i,x in enumerate(expected[1:],1):settle(store,x['submission_id'],status='passed' if i==1 else 'error',manifest=x['manifest_sha256'])
    apply(c,M636.read_text());apply(c,M636.read_text())
    assert [x['generation_floor'] for x in store.scoring_policy_history(ROUND)['revisions']]==[0,0]
    r=store.get_round(ROUND)
    assert r['participants'][:18]==participants
    assert [p['submission_id'] for p in r['participants'][18:]]==[expected[1]['submission_id']]
    assert store.get_submission(expected[1]['submission_id'])['status']=='frozen'
    assert all(store.get_submission(x['submission_id'])['status']=='rejected' for i,x in enumerate(expected) if i!=1)
    assert store.list_runs(ROUND)==[]
    with c.cursor() as q:
        q.execute('SELECT operator_paused,pause_reason FROM public.lab_arena_restart_claim_control WHERE singleton')
        assert q.fetchone()==(True,'await_reviewed_recovery633_before_publication')
    for x in expected:
        after=store.list_ledger(submission_id=x['submission_id'])
        assert after[:len(before[x['submission_id']])]==before[x['submission_id']]
        assert sum(l['entry_kind']=='reservation' for l in after)==4


@pytest.mark.parametrize('fault',['source','ledger','round','hold','guard','preserved'])
def test_initial_receipt_rejects_drift_without_changes(context,fault):
    store,c,sql,expected,participants,_=context
    if fault=='source':sql=sql.replace(expected[0]['source_sha256'],'sha256:'+'0'*64)
    elif fault=='ledger':sql=sql.replace(json.dumps(expected[0]['ledger_ids'],separators=(',',':')), '[0]')
    else:
        with c.cursor() as q:
            if fault=='round':q.execute("UPDATE public.lab_arena_rounds SET updated_at=clock_timestamp() WHERE round_id=%s",(ROUND,))
            elif fault=='hold':q.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE WHERE singleton')
            elif fault=='guard':acquire(c)
            else:q.execute("UPDATE public.lab_arena_submissions SET updated_at=clock_timestamp() WHERE submission_id=%s",(participants[1]['submission_id'],))
    with pytest.raises(Exception,match='differ'):apply(c,sql)
    assert all(store.get_submission(x['submission_id'])['status']=='rejected' for x in expected)


def test_recovery_hint_expiry_fails_closed():
    now=datetime.now(timezone.utc)
    row={'round_id':ROUND,'status':'accepted','code_review_status':'error','code_review_attempts':3,'code_review_doc':{
        'error_code':'review_response_invalid','review_invalid_attempts':3,'nov06_late_review':{
            'request_id':'nov06-late-review635-v1','expires_at':now.isoformat()}}}
    assert not code_review_policy.nov06_late_review_available(row,now=now)
    assert code_review_policy.nov06_late_review_available(row,now=now-timedelta(microseconds=1))
    row['code_review_attempts']=4
    assert not code_review_policy.nov06_late_review_available(row,now=now-timedelta(seconds=1))


def test_native_full_source_runtime_review_uses_fourth_sol_and_stored_owner(context):
    from tests.lab_arena.refusal_review_fallback_test import reviewer_for,router_receipt
    from tests.lab_arena.test_lab_arena_code_review_runtime import ReviewTransport
    from tests.lab_arena.research_control_code_review_test import _assessment,_response
    store,c,sql,expected,_,payload=context
    sid=expected[0]['submission_id'];row=store.get_submission(sid)
    assert store.get_submission_credential(sid,row['miner_hotkey'],'openrouter') is None
    apply(c,sql)
    assert store.get_submission_credential(sid,row['miner_hotkey'],'openrouter')['submission_id']==sid
    def response(parameters):
        assert parameters['model']==ALTERNATE
        assert parameters['provider']['only']==['azure']
        prepared=code_review.prepare_request(payload,model=ALTERNATE,research_control=True,
            context_window_tokens=code_review.FALLBACK_CONTEXT_WINDOW_TOKENS)
        reply=_response(prepared,_assessment(prepared))
        reply['usage']={'cost':'0.001234'}
        router_receipt(reply,parameters,prompt_tokens=1000)
        return {'body':reply}
    transport=ReviewTransport(response);reviewer,_=reviewer_for(payload,transport,store)
    assert reviewer.review(store.get_submission(sid))['status']=='passed'
    row=store.get_submission(sid)
    assert row['code_review_attempts']==4
    assert row['code_review_doc']['research_control']['source_manifest_sha256']==expected[0]['manifest_sha256']
    assert row['code_review_doc']['cost_microusd']==1234
    assert len(transport.sent)==1


def test_expired_fourth_is_uncertain_then_excluded_without_fifth(context):
    store,c,sql,expected,_,_=context
    apply(c,sql)
    sid=expected[0]['submission_id'];miner=store.get_submission(sid)['miner_hotkey']
    assert begin(store,sid,miner,ALTERNATE)['status']=='claimed'
    with c.cursor() as q:
        q.execute("UPDATE public.lab_arena_submissions SET code_review_started_at=clock_timestamp()-interval '20 minutes',code_review_expires_at=clock_timestamp()-interval '1 second' WHERE submission_id=%s",(sid,))
    assert begin(store,sid,miner,ALTERNATE)['status']=='recovered'
    assert begin(store,sid,miner,ALTERNATE)['status']=='exhausted'
    row=store.get_submission(sid)
    assert row['code_review_status']=='error' and row['code_review_attempts']==4
    assert row['code_review_doc']['cost_status']=='uncertain'
    for x in expected[1:]:settle(store,x['submission_id'])
    apply(c,M636.read_text())
    assert store.get_submission(sid)['status']=='rejected'
    assert sum(l['entry_kind']=='reservation' for l in store.list_ledger(submission_id=sid))==4


def test_late_success_becomes_accounted_error_and_unspent_window_cannot_extend(context):
    store,c,sql,expected,_,_=context
    apply(c,sql)
    sid=expected[0]['submission_id'];row=store.get_submission(sid);token=new_lease_token()
    assert begin(store,sid,row['miner_hotkey'],ALTERNATE,token)['status']=='claimed'
    # Time travel only this disposable append-only receipt, keeping its metadata
    # consistent, to prove the real finish RPC after the30-minute window.
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        q.execute("UPDATE public.lab_arena_scoring_supersessions SET created_at=clock_timestamp()-interval '31 minutes',recovery_doc=jsonb_set(recovery_doc,'{expires_at}',to_jsonb((clock_timestamp()-interval '1 minute')::TEXT)) WHERE request_id='nov06-late-review635-v1'")
        q.execute("UPDATE public.lab_arena_ledger SET created_at=clock_timestamp()-interval '30 minutes' WHERE submission_id=%s AND call_identity=(SELECT call_identity FROM public.lab_arena_ledger WHERE submission_id=%s AND entry_kind='reservation' AND entry_doc->'attempt'='4'::JSONB)",(sid,sid))
        q.execute("SET session_replication_role='origin'")
    doc={**_document(status='passed'),'model':ALTERNATE,'file_count':2,'source_bytes':1000}
    doc['research_control'].update(source_archive_sha256=row['source_archive_sha256'],source_manifest_sha256=expected[0]['manifest_sha256'])
    result=store.finish_submission_review(sid,row['miner_hotkey'],token,'passed',doc,4321)
    assert result['status']=='deadline'
    row=store.get_submission(sid)
    assert row['code_review_doc']['error_code']=='code_review_deadline_exceeded'
    assert row['code_review_doc']['cost_microusd']==4321
    assert row['code_review_doc']['cost_status']=='settled'
    before=store.list_ledger(submission_id=sid)
    assert begin(store,sid,row['miner_hotkey'],ALTERNATE)['status']=='deadline'
    assert store.list_ledger(submission_id=sid)==before
    apply(c,sql)  # Reapply cannot renew the receipt window.
    for x in expected[1:]:
        r=store.get_submission(x['submission_id'])
        assert begin(store,x['submission_id'],r['miner_hotkey'],ALTERNATE)['status']=='deadline'
    apply(c,M636.read_text())
    assert all(store.get_submission(x['submission_id'])['status']=='rejected' for x in expected)
    assert len(store.get_round(ROUND)['participants'])==18


@pytest.mark.parametrize('value',['missing',None,True,2.5,'2'])
def test_finalizer_requires_typed_complete_evidence_count(context,value):
    store,c,sql,expected,_,_=context
    apply(c,sql)
    for i,x in enumerate(expected):settle(store,x['submission_id'],status='passed' if i==0 else 'error',manifest=x['manifest_sha256'])
    with c.cursor() as q:
        if value=='missing':
            q.execute("UPDATE public.lab_arena_submissions SET code_review_doc=code_review_doc#-'{research_control,evidence_count}' WHERE submission_id=%s",(expected[0]['submission_id'],))
        else:
            q.execute("UPDATE public.lab_arena_submissions SET code_review_doc=jsonb_set(code_review_doc,'{research_control,evidence_count}',%s::JSONB) WHERE submission_id=%s",(json.dumps(value),expected[0]['submission_id']))
    with pytest.raises(Exception,match='passing full-source proof differs'):apply(c,M636.read_text())
    assert len(store.get_round(ROUND)['participants'])==18


@pytest.mark.parametrize('fault',['source','preserved','round','hold','guard'])
def test_granted_receipt_cannot_spend_after_bound_state_changes(context,fault):
    store,c,sql,expected,participants,_=context
    apply(c,sql)
    sid=expected[0]['submission_id'];row=store.get_submission(sid)
    before=store.list_ledger(submission_id=sid)
    with c.cursor() as q:
        if fault=='source':q.execute("UPDATE public.lab_arena_submissions SET source_size_bytes=source_size_bytes+1,submission_doc=jsonb_set(submission_doc,'{source_size_bytes}',to_jsonb(source_size_bytes+1)) WHERE submission_id=%s",(sid,))
        elif fault=='preserved':q.execute("UPDATE public.lab_arena_submissions SET updated_at=clock_timestamp() WHERE submission_id=%s",(participants[1]['submission_id'],))
        elif fault=='round':
            q.execute("SET session_replication_role='replica'")
            q.execute("UPDATE public.lab_arena_rounds SET benchmark_ref=benchmark_ref||'.changed' WHERE round_id=%s",(ROUND,))
            q.execute("SET session_replication_role='origin'")
        elif fault=='hold':q.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE WHERE singleton')
        else:acquire(c)
    assert begin(store,sid,row['miner_hotkey'],ALTERNATE)['status'] in {'exhausted','paused','round_closed','deadline'}
    assert store.list_ledger(submission_id=sid)==before


def test_helper_drift_is_rejected_and_reopen_gate_remains_exact(context):
    store,c,sql,expected,_,_=context
    apply(c,sql)
    with c.cursor() as q:
        q.execute("SELECT pg_get_functiondef('public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)'::regprocedure)")
        helper=q.fetchone()[0]
        q.execute(helper.replace(' SELECT h.created_at',' SELECT /* drift */ h.created_at',1))
    with pytest.raises(Exception,match='helper differs'):apply(c,sql)
    with c.cursor() as q:q.execute(helper)
    sid=expected[0]['submission_id']
    with c.cursor() as q:q.execute("UPDATE public.lab_arena_submissions SET status='rejected' WHERE submission_id=%s",(sid,))
    with pytest.raises(Exception,match='cannot be reopened'):
        with c.cursor() as q:q.execute("UPDATE public.lab_arena_submissions SET status='accepted' WHERE submission_id=%s",(sid,))


@pytest.mark.parametrize('before',[True,False])
def test_history633_and_late635_compose_in_either_order(context,before):
    store,c,sql,_,_,_=context
    text=(SCRIPTS/'633-arena-2026-11-05-approved-infrastructure-recovery.sql').read_text()
    history='DO $history633$'+text.split('DO $history633$')[1].split('$history633$;')[0]+'$history633$;'
    if before:apply(c,history)
    apply(c,sql)
    if not before:apply(c,history)
    apply(c,sql);apply(c,history)
    assert [x['generation_floor'] for x in store.scoring_policy_history(ROUND)['revisions']]==[0]
    with c.cursor() as q:
        q.execute("SELECT encode(extensions.digest(pg_get_functiondef('public.lab_arena_scoring_policy_history_v1(text)'::regprocedure),'sha256'),'hex')")
        assert q.fetchone()[0]=='e56c877e3c2ba5440b2fc7b55caa65b2e94935c5117dddd12ed1320c9e3f4a45'


def test_native_canonical_guard_allows_inflight_completion_but_no_new_claim(context):
    store,c,sql,expected,_,_=context
    apply(c,sql)
    sid=expected[0]['submission_id'];row=store.get_submission(sid);token=new_lease_token()
    assert begin(store,sid,row['miner_hotkey'],ALTERNATE,token)['status']=='claimed'
    acquire(c)
    other=expected[1]['submission_id'];miner=store.get_submission(other)['miner_hotkey']
    assert begin(store,other,miner,ALTERNATE)['status']=='paused'
    doc={**_document(status='passed'),'model':ALTERNATE,'file_count':2,'source_bytes':1000}
    doc['research_control'].update(source_archive_sha256=row['source_archive_sha256'],source_manifest_sha256=expected[0]['manifest_sha256'])
    assert store.finish_submission_review(sid,row['miner_hotkey'],token,'passed',doc,4321)['status']=='passed'
    assert store.get_submission(sid)['code_review_status']=='passed'
