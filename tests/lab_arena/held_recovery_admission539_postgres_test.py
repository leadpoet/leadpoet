"""Held recovered execution stops new liabilities while admitted calls drain."""
from pathlib import Path
import json
import os
import pytest
from tests.lab_arena import restore_recorded_results538_postgres_test as previous
recovery=previous.recovery
SQL=Path(__file__).parents[2]/'scripts/539-lab-arena-held-recovery-provider-admission.sql'
SIG='public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
TOKEN='d'*64

@pytest.fixture(scope='module')
def database():
    gen=previous.database.__wrapped__();db=next(gen)
    try:
        with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure",(SIG,));before=cur.fetchone()[0]
            cur.execute(SQL.read_text());cur.execute(SQL.read_text())
            cur.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure",(SIG,));assert cur.fetchone()[0]==before
            cur.execute("SELECT jsonb_build_object('sha256',encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex'),'owner',pg_get_userbyid(p.proowner),'acl',p.proacl::text,'definer',p.prosecdef,'volatility',p.provolatile,'config',p.proconfig,'language',l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=%s::regprocedure",(SIG,));catalog={SIG:cur.fetchone()[0]}
            if os.environ.get('ARENA_FUNCTION_CATALOG_OUT'):
                Path(os.environ['ARENA_FUNCTION_CATALOG_OUT']).write_text(json.dumps(catalog,indent=2)+'\n')
        yield db
    finally:gen.close()


def prepared(db, *, ordinary=False,score=False,held=True,wrong_floor=False,wrong_config=False):
    store,row,cfg,items=recovery.seed(db)
    with db[0].connect(**db[1]) as conn,conn.cursor() as cur:
        if not ordinary:recovery.invoke(cur,row,cfg,items)
        cur.execute('SET LOCAL session_replication_role=replica')
        if ordinary:cur.execute("UPDATE public.lab_arena_rounds SET status='stage2',configuration_doc=%s::jsonb WHERE round_id=%s",(json.dumps(cfg),row['round_id']))
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(row['round_id'],));fresh=cur.fetchone()[0]
        if score:
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,lease_token_hash=%s,lease_generation=1,lease_expires_at=clock_timestamp()+interval '1 hour' WHERE round_id=%s AND kind='score' AND stage=1 AND scoring_generation_floor=%s RETURNING run_id",(recovery.base.company.original.hotkey('runner'),TOKEN,row['round_id'],fresh['scoring_generation_floor']))
        else:
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,lease_token_hash=%s,lease_generation=1,lease_expires_at=clock_timestamp()+interval '1 hour',stage_generation=%s WHERE round_id=%s AND kind='execute' AND stage=2 AND icp_position=1 AND attempt=%s RETURNING run_id",(recovery.base.company.original.hotkey('runner'),TOKEN,fresh['stage_generation'],row['round_id'],1 if ordinary else 2))
        run=cur.fetchone()[0]
        if wrong_floor:cur.execute('UPDATE public.lab_arena_rounds SET scoring_generation_floor=scoring_generation_floor+1 WHERE round_id=%s',(row['round_id'],))
        if wrong_config:cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc||'{\"fixture_config_difference\":true}'::jsonb WHERE round_id=%s",(row['round_id'],))
        cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=%s WHERE singleton',(held,))
        cur.execute('SET LOCAL session_replication_role=origin')
    return store,run


def reserve(store,run,marker='a'):
    return store.reserve_call(run_id=run,lease_token_hash=TOKEN,call_identity='sha256:'+marker*64,
      operation_id='openrouter.chat',provider='openrouter',funding_source=store.provider_funding(run,'openrouter')['funding_source'],
      amount_microusd=100,call_doc={'model':'fixture'},lease_ttl_seconds=3600)


@pytest.mark.parametrize('case',['held','unheld','ordinary','score','wrong_floor','wrong_config'])
def test_only_held_current_recovered_execution_stops_fresh_reservations(database,case):
    store,run=prepared(database,ordinary=case=='ordinary',score=case=='score',held=case!='unheld',wrong_floor=case=='wrong_floor',wrong_config=case=='wrong_config')
    try:
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(run,));before=cur.fetchone()[0]
            cur.execute('SELECT count(*) FROM public.lab_arena_ledger');count=cur.fetchone()[0]
        result=reserve(store,run)
        assert result['status']==('stale' if case=='held' else 'reserved'),result
        if case=='held':
            with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
                cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(run,));assert cur.fetchone()[0]==before
                cur.execute('SELECT count(*) FROM public.lab_arena_ledger');assert cur.fetchone()==(count,)
    finally:store.close()


@pytest.mark.parametrize('late',[False,True])
def test_preexisting_reservation_replays_dispatches_and_settles_under_hold(database,late):
    store,run=prepared(database,held=False)
    try:
        assert reserve(store,run)['status']=='reserved'
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton')
        replay=reserve(store,run)
        assert replay['status']=='reserved' and replay['idempotent'] is True
        assert reserve(store,run,'b')['status']=='stale'
        identity='sha256:'+'a'*64
        assert store.mark_dispatched(run_id=run,lease_token_hash=TOKEN,call_identity=identity)['status']=='dispatched'
        if late:
            # Real canonical cancellation in the disposable database supplies
            # the exact closed-run uncertainty proof for late reconciliation.
            with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
                cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
            assert store.cancel_round(run.split(':')[0],'test_owner_stop')['status']=='cancelled'
            with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
                cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton')
        terminal={'status':200,'headers':{},'body_b64':'e30='}
        result=store.settle_call(run_id=run,lease_token_hash=TOKEN,call_identity=identity,actual_microusd=75,terminal_response=terminal,lease_ttl_seconds=3600)
        assert result['status']=='settled',result
        assert store.settle_call(run_id=run,lease_token_hash=TOKEN,call_identity=identity,actual_microusd=75,terminal_response=terminal,lease_ttl_seconds=3600)['idempotent'] is True
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SELECT entry_kind,amount_microusd FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id',(identity,))
            rows=cur.fetchall();assert [r[0] for r in rows]==(['reservation','dispatch','uncertain','settlement'] if late else ['reservation','dispatch','settlement']) and rows[-1][1]==75
    finally:store.close()


@pytest.mark.parametrize('fault',['grant','source'])
def test_exact_patch_fails_closed_on_catalog_drift(database,fault):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)',(SIG,));original=cur.fetchone()[0]
        if fault=='grant':cur.execute('GRANT EXECUTE ON FUNCTION '+SIG+' TO anon')
        else:cur.execute(original.replace('lab_arena_reserve_input_invalid','unreviewed reserve input'))
        conn.commit()
        try:
            with pytest.raises(database[0].Error):cur.execute(SQL.read_text())
        finally:
            conn.rollback()
            if fault=='grant':cur.execute('REVOKE EXECUTE ON FUNCTION '+SIG+' FROM anon')
            else:cur.execute(original)


def test_settled_boundary_restoration_still_reaches_ordinary_publication(database,tmp_path,monkeypatch):
    previous.test_archived_authority_restores_through_ordinary_aggregate_and_publication(database,tmp_path,monkeypatch,True)
