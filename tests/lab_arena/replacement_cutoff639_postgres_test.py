"""Prospective cutoff639 on real admission/review locks and durable reservations."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json

import pytest
from psycopg2.extras import Json

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.nov06_late_review635_postgres_test import database as predecessor_database
from tests.lab_arena.queued_submission_replacement_postgres_test import _config, _doc, _accept
from tests.lab_arena.test_lab_arena_source_migration_postgres import _hotkey
from tests.lab_arena.refusal_review_fallback599_postgres_test import begin, PRIMARY
from tests.postgres_migration_harness import SCRIPTS

MIGRATION = SCRIPTS/'642-lab-arena-prospective-replacement-cutoff.sql'
SIGNATURES = (
 'public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)',
 'public.lab_arena_accept_submission_with_credentials(text,text,text,jsonb)',
 'public.lab_arena_accept_submission_source_with_credentials(text,text,text,jsonb,text,text)',
 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)',
)


def create(store, suffix, *, minutes=20):
    rid='arena-2098-07-01-'+suffix
    store.create_round(rid, _config(rid, cutoff_after=timedelta(minutes=minutes)))
    return rid, _hotkey('cutoff639-'+suffix)


def first(store, rid, miner, sid):
    assert store.register_submission(rid,sid,miner,_doc(rid,sid,123))['status']=='registered'
    assert _accept(store,rid,sid,miner)['status']=='ok'


@pytest.fixture(scope='module')
def database():
    fixture=predecessor_database.__wrapped__()
    try:
        pg,dsn,_=next(fixture)
        with pg.connect(**dsn) as c,c.cursor() as q:
            text=(SCRIPTS/'635-arena-2026-11-06-late-review-recovery.sql').read_text()
            # Install the exact production function bodies, without dated grants.
            q.execute(text.split('DO $recover635$')[0]+'DO $restore635$'+text.split('DO $restore635$',1)[1])
        yield pg,dsn
    finally:fixture.close()


@pytest.fixture(scope='module')
def context(database):
    pg,dsn=database
    store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
    c=pg.connect(**dsn);c.autocommit=True
    rid,miner=create(store,'backfill')
    first(store,rid,miner,'cutoff639-old')
    assert store.register_submission(rid,'cutoff639-before',miner,_doc(rid,'cutoff639-before',124))['status']=='replacement_closed'
    reviewed,rminer=create(store,'reviewed')
    first(store,reviewed,rminer,'cutoff639-reviewed')
    assert begin(store,'cutoff639-reviewed',rminer,PRIMARY)['status']=='claimed'
    frozen,_=create(store,'frozen')
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        q.execute("UPDATE public.lab_arena_rounds SET status='committed',status_generation=1,participants='[]',benchmark_ref='fixture/benchmark' WHERE round_id=%s",(frozen,))
        q.execute("SET session_replication_role='origin'")
        q.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",(frozen,));before=q.fetchone()[0]
        q.execute("SELECT jsonb_object_agg(oid::TEXT,pg_get_functiondef(oid)) FROM pg_proc WHERE oid IN('public.lab_arena_rounds_write_once_v1()'::regprocedure,'public.lab_arena_scoring_policy_history_v1(text)'::regprocedure)");unchanged=q.fetchone()[0]
        q.execute(MIGRATION.read_text());q.execute(MIGRATION.read_text())
        q.execute("SELECT to_jsonb(r)-'replacement_freeze_seconds' FROM public.lab_arena_rounds r WHERE round_id=%s",(frozen,));assert q.fetchone()[0]==before
        q.execute("SELECT jsonb_object_agg(oid::TEXT,pg_get_functiondef(oid)) FROM pg_proc WHERE oid IN('public.lab_arena_rounds_write_once_v1()'::regprocedure,'public.lab_arena_scoring_policy_history_v1(text)'::regprocedure)");assert q.fetchone()[0]==unchanged
    assert store.get_round(rid)['replacement_freeze_seconds']==600
    assert store.get_round(reviewed)['replacement_freeze_seconds'] is None
    assert store.get_round(frozen)['replacement_freeze_seconds'] is None
    # Unchanged258 fields keep existing RPC clients compatible; new code probes
    # the explicit capability before it selects the additive thin column.
    schema=store.submission_replacement_schema(require_per_round_cutoff=True)
    assert schema['replacement_freeze_seconds']==3600 and schema['max_replacement_attempts']==1
    assert schema['default_replacement_freeze_seconds']==600
    assert store.register_submission(rid,'cutoff639-after',miner,_doc(rid,'cutoff639-after',124))['status']=='registered'
    assert _accept(store,rid,'cutoff639-after',miner)['status']=='ok'
    yield store,c
    store.close();c.close()


@contextmanager
def clock(c, moment):
    """Pin only disposable RPC clocks to prove each exact SQL boundary."""
    original=[]
    with c.cursor() as q:
        for signature in SIGNATURES:
            q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(signature,))
            definition=q.fetchone()[0];original.append(definition)
            q.execute(definition.replace('pg_catalog.clock_timestamp()', "TIMESTAMPTZ '%s'" % moment.isoformat()))
    try:yield
    finally:
        with c.cursor() as q:
            for definition in original:q.execute(definition)


def test_backfill_future_default_and_frozen_field_guard(context):
    store,c=context
    rid,miner=create(store,'default')
    row=store.get_round(rid)
    assert row['replacement_freeze_seconds']==600
    assert row['configuration_doc']['schedule']['submission_cutoff']
    thin = store.list_rounds(columns='round_id,'+store.replacement_cutoff_columns+'status')
    assert next(r for r in thin if r['round_id']==rid)['replacement_freeze_seconds']==600
    with c.cursor() as q:
        with pytest.raises(Exception,match='lab_arena_replacement_policy_frozen'):
            q.execute('UPDATE public.lab_arena_rounds SET replacement_freeze_seconds=3600 WHERE round_id=%s',(rid,))
        q.execute('SELECT round_id,replacement_freeze_seconds FROM public.lab_arena_rounds ORDER BY round_id')
        assert all(value in (None,600) for _,value in q.fetchall())


@pytest.mark.parametrize('delta',[-1,0,1])
def test_native_register_finalize_and_review_exact_2350(context,delta):
    store,c=context
    rid,miner=create(store,'edge'+str(delta).replace('-','m'),minutes=180)
    first(store,rid,miner,'cutoff639-first-'+str(delta))
    candidate='cutoff639-final-'+str(delta)
    assert store.register_submission(rid,candidate,miner,_doc(rid,candidate,124))['status']=='registered'
    cutoff=datetime.fromisoformat(store.get_round(rid)['configuration_doc']['schedule']['submission_cutoff'].replace('Z','+00:00'))
    moment=cutoff-timedelta(seconds=600)+timedelta(microseconds=delta)
    other=_hotkey('cutoff639-other'+str(delta))
    first(store,rid,other,'cutoff639-other-'+str(delta))
    with clock(c,moment):
        registered=store.register_submission(rid,'cutoff639-register-'+str(delta),other,_doc(rid,'cutoff639-register-'+str(delta),125))
        assert registered['status']==('registered' if delta<0 else 'replacement_closed')
        if delta<0:
            assert _accept(store,rid,candidate,miner)['status']=='ok'
        else:
                assert _accept(store,rid,candidate,miner)['status']=='replacement_closed'
        sid='cutoff639-other-'+str(delta)
        # An uploading replacement also defers the original review, so terminally
        # reject it first; the durable allowance remains consumed.
        if delta<0:
            assert store.update_submission(rid,'cutoff639-register-'+str(delta),'uploading','rejected',{'rejection_rule':'fixture_invalid_source'})['status']=='ok'
        result=begin(store,sid,other,PRIMARY)
        assert result['status']==('deferred' if delta<0 else 'claimed')
        if delta<0:assert store.list_ledger(submission_id=sid)==[]


def test_source_finalize_rechecks_cutoff_and_prior_evaluation(context):
    store,c=context
    rid,miner=create(store,'source',minutes=180)
    first(store,rid,miner,'cutoff639-source-old')
    sid='cutoff639-source-new'
    assert store.register_submission(rid,sid,miner,_doc(rid,sid,124))['status']=='registered'
    cutoff=datetime.fromisoformat(store.get_round(rid)['configuration_doc']['schedule']['submission_cutoff'].replace('Z','+00:00'))
    with clock(c,cutoff-timedelta(seconds=600)):
        assert store.accept_submission_source_with_credentials(rid,sid,miner,{'openrouter':'dGVzdA==','deepline':'dGVzdA=='},'sha256:'+'1'*64,'sha256:'+'2'*64)['status']=='replacement_closed'
    # A benchmark proves evaluation started. It blocks replacements even when
    # the wall clock remains before23:50; no source/credential changes occur.
    with c.cursor() as q:
        q.execute("SET session_replication_role='replica'")
        q.execute("UPDATE public.lab_arena_rounds SET benchmark_ref='fixture/evaluating' WHERE round_id=%s",(rid,))
        q.execute("SET session_replication_role='origin'")
    with pytest.raises(ArenaStoreError,match='replacement_ineligible'):_accept(store,rid,sid,miner)
    assert store.get_submission('cutoff639-source-old')['status']=='accepted'


def test_source_finalize_before_cutoff_and_midnight_first_intake(context):
    store,c=context
    rid,miner=create(store,'sourcepass',minutes=180)
    first(store,rid,miner,'cutoff639-pass-old')
    sid='cutoff639-pass-new'
    assert store.register_submission(rid,sid,miner,_doc(rid,sid,124))['status']=='registered'
    cutoff=datetime.fromisoformat(store.get_round(rid)['configuration_doc']['schedule']['submission_cutoff'].replace('Z','+00:00'))
    with clock(c,cutoff-timedelta(seconds=600,microseconds=1)):
        assert store.accept_submission_source_with_credentials(rid,sid,miner,{'openrouter':'dGVzdA==','deepline':'dGVzdA=='},'sha256:'+'3'*64,'sha256:'+'4'*64)['status']=='ok'
    assert store.get_submission(sid)['source_archive_sha256']=='sha256:'+'3'*64
    for delta in (-1,0,1):
        other=_hotkey('cutoff639-midnight'+str(delta));new='cutoff639-midnight'+str(delta)
        with clock(c,cutoff+timedelta(microseconds=delta)):
            assert store.register_submission(rid,new,other,_doc(rid,new,125))['status']==('registered' if delta<0 else 'window_closed')


def test_function_drift_rejected_without_overwriting_it(context):
    store,c=context
    signature=SIGNATURES[0]
    with c.cursor() as q:
        q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(signature,));original=q.fetchone()[0]
        drift=original.replace('DECLARE','DECLARE\n-- fixture preimage drift',1)
        q.execute(drift)
        try:
            with pytest.raises(Exception,match='preimage differs'):q.execute(MIGRATION.read_text())
            q.execute('ROLLBACK')
            q.execute('SELECT pg_get_functiondef(%s::regprocedure)',(signature,));assert 'fixture preimage drift' in q.fetchone()[0]
        finally:q.execute(original)


def test_migration_blocks_unfinalized_ninth_receipt(context):
    store,c=context
    rid,_=create(store,'unfinalized')
    with c.cursor() as q:
        q.execute('BEGIN')
        try:
            q.execute("INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc) SELECT round_id,1,'nov06-ninth-review637-v1',%s,'fixture',txid_current(),pg_backend_pid(),to_jsonb(r),to_jsonb(r),'{}','{}' FROM public.lab_arena_rounds r WHERE round_id=%s",('sha256:'+'5'*64,rid))
            with pytest.raises(Exception,match='requires completed November 6 recovery'):q.execute(MIGRATION.read_text())
        finally:q.execute('ROLLBACK')


def test_migration_blocks_ninth_target_even_before_its_grant(context):
    store,c=context
    rid,miner=create(store,'pregrant')
    sid='sub-43a929eb0e2e487f68014f48e7d80e6b'
    with c.cursor() as q:
        q.execute('BEGIN')
        try:
            q.execute('SELECT public.lab_arena_register_submission(%s,%s,%s,%s)',(rid,sid,miner,Json(_doc(rid,sid,123))))
            with pytest.raises(Exception,match='requires completed ninth recovery'):q.execute(MIGRATION.read_text())
        finally:q.execute('ROLLBACK')


def test_concurrent_single_allowance_and_invalid_attempt_remains_spent(context,database):
    store,c=context;pg,dsn=database
    rid,miner=create(store,'race')
    first(store,rid,miner,'cutoff639-race-old')
    def register(index):
        client=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
        try:return client.register_submission(rid,'cutoff639-race-'+str(index),miner,_doc(rid,'cutoff639-race-'+str(index),124+index))
        finally:client.close()
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(register,(1,2)))
    assert sorted(x['status'] for x in results)==['registered','replacement_limit_reached']
    sid=next(x['submission_id'] for x in results if x['status']=='registered')
    assert store.update_submission(rid,sid,'uploading','rejected',{'rejection_rule':'fixture_invalid_source'})['status']=='ok'
    assert register(3)['status']=='replacement_limit_reached'
    assert store.get_submission('cutoff639-race-old')['status']=='accepted'
    assert store.list_ledger(submission_id='cutoff639-race-old')==[]
    assert store.list_ledger(submission_id=sid)==[]
