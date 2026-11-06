"""Expiring owner routing changes new claims only, including handoff alternatives."""
from copy import deepcopy
import json
from pathlib import Path
import uuid

import pytest
from tests.lab_arena import nov05_b66_sixth639_postgres_test as native

prior=native.recovery
ROUND=prior.ROUND
PRIMARY=native.hotkey('645-primary')
OTHER=native.hotkey('645-other')
SQL=prior.source.SCRIPTS/'645-lab-arena-expiring-round-claim-routing.sql'
database=native.database
FIELDS=['new_claim_round_id','new_claim_allowed_runner_hotkey','new_claim_until','new_claim_actor_ref']


def route(p,*,round_id=ROUND,runner=PRIMARY,until='2099-11-06T09:35:00Z'):
    with p.conn.cursor()as q:
        q.execute('UPDATE public.lab_arena_restart_claim_control SET new_claim_round_id=%s,new_claim_allowed_runner_hotkey=%s,new_claim_until=%s,new_claim_actor_ref=%s WHERE singleton',(round_id,runner,until,'fixture645'))
    p.conn.commit()


def claim(p,runner=PRIMARY,request=None):
    request=request or uuid.uuid4().hex
    with p.conn.cursor()as q:
        q.execute('SET ROLE lab_arena_service')
        q.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,1500)',(ROUND,runner,request,'sha256:'+'b'*64,prior.TOKEN))
        result=q.fetchone()[0];q.execute('RESET ROLE')
    p.conn.commit();return result


@pytest.fixture
def prepared(database):
    pg,dsn,definitions=database
    with pg.connect(**dsn)as c,c.cursor()as q:
        for d in definitions.values():q.execute(d)
        q.execute('UPDATE public.lab_arena_restart_claim_control SET '+','.join(f+'=NULL'for f in FIELDS)+' WHERE singleton')
    p=prior.seed((pg,dsn))
    try:
        with p.conn.cursor()as q:
            cfg=prior.scalar(q,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))
            cfg['runner_hotkeys']=[PRIMARY,OTHER]
            q.execute('SET LOCAL session_replication_role=replica')
            q.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',(json.dumps(cfg),ROUND))
            q.execute('SET LOCAL session_replication_role=origin')
        p.conn.commit();p.sql=prior.render(p);prior.apply(p)
        yield p
    finally:
        p.conn.rollback()
        with p.conn.cursor()as q:q.execute('UPDATE public.lab_arena_restart_claim_control SET '+','.join(f+'=NULL'for f in FIELDS)+' WHERE singleton')
        p.conn.commit();p.store.close();p.conn.close()


def test_default_null_allows_both_and_active_route_blocks_only_new_other_claims(prepared):
    p=prepared
    assert claim(p,OTHER)['status']=='leased'
    assert claim(p,PRIMARY)['status']=='leased'
    route(p)
    before=prior.snapshot(p)
    assert claim(p,OTHER)=={'status':'paused'}
    assert prior.snapshot(p)==before
    assert claim(p,PRIMARY)['status']=='leased'


@pytest.mark.parametrize('mode',['expired','other_round'])
def test_expiry_and_nonmatching_round_have_no_effect(prepared,mode):
    route(prepared,round_id='arena-2026-11-06'if mode=='other_round'else ROUND,
          until='2000-01-01T00:00:00Z'if mode=='expired'else'2099-11-06T09:35:00Z')
    assert claim(prepared,OTHER)['status']=='leased'


def test_existing_other_replay_provider_and_completion_remain_valid(prepared):
    p=prepared;request=uuid.uuid4().hex;lease=claim(p,OTHER,request)
    route(p)
    assert claim(p,OTHER,request)==lease
    prior.bill(p,lease)
    assert prior.complete(p,lease)['status']=='accepted'
    assert claim(p,OTHER)=={'status':'paused'}


def test_primary_self_retry_ignores_routed_out_live_alternative(prepared):
    p=prepared
    other=claim(p,OTHER)
    first=claim(p,PRIMARY)
    # Isolate one ordinary first-attempt retry; all creation uses native RPCs.
    while first['attempt']!=1:
        assert prior.complete(p,first)['status']=='accepted'
        first=claim(p,PRIMARY)
    assert prior.complete(p,first,cause='provider_error')['status']=='failed'
    successor=first['assignment_id']+':2'
    with p.conn.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE round_id=%s AND status='pending' AND run_id<>%s",(ROUND,successor))
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    assert claim(p,PRIMARY)['status']!='leased'
    route(p)
    lease=claim(p,PRIMARY)
    assert lease['status']=='leased'and lease['run_id']==successor
    assert p.store.get_run(other['run_id'])['status']=='leased'


@pytest.mark.parametrize('field,value',[
    ('new_claim_round_id',ROUND),('new_claim_allowed_runner_hotkey',PRIMARY),
    ('new_claim_until','2099-11-06'),('new_claim_actor_ref','x')])
def test_partial_control_tuple_is_rejected(prepared,field,value):
    p=prepared
    with p.conn.cursor()as q:
        with pytest.raises(p.pg.errors.CheckViolation):
            q.execute('UPDATE public.lab_arena_restart_claim_control SET '+field+'=%s WHERE singleton',(value,))
    p.conn.rollback()


def test_owner_only_control_and_trigger_security(prepared):
    p=prepared
    with p.conn.cursor()as q:
        q.execute('SET ROLE lab_arena_service')
        with pytest.raises(p.pg.errors.InsufficientPrivilege):
            q.execute('UPDATE public.lab_arena_restart_claim_control SET new_claim_actor_ref=NULL WHERE singleton')
    p.conn.rollback()
    with p.conn.cursor()as q:
        identity=prior.scalar(q,"SELECT jsonb_build_array(pg_get_userbyid(proowner),proacl::text,prosecdef,provolatile,proconfig) FROM pg_proc WHERE oid='public.lab_arena_restart_claim_gate_v1()'::regprocedure")
    assert identity==['lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',True,'v',['search_path=pg_catalog, public']]


def test_trigger_backstop_blocks_old_claim_body_but_allows_nov6(prepared):
    p=prepared;route(p)
    pending=next(r for r in p.store.list_runs(ROUND,kind='execute')if r['status']=='pending')
    with p.conn.cursor()as q:
        with pytest.raises(p.pg.errors.ObjectNotInPrerequisiteState,match='claims_paused'):
            q.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s WHERE run_id=%s",(OTHER,pending['run_id']))
    p.conn.rollback()
    # A direct trigger call on a disposable table proves a different round's
    # pending-to-leased update is unchanged even for the routed-out runner.
    with p.conn.cursor()as q:
        q.execute('CREATE TEMP TABLE route645_trigger_control(status text,round_id text,runner_hotkey text)')
        q.execute('CREATE TRIGGER route645 BEFORE UPDATE OF status ON route645_trigger_control FOR EACH ROW EXECUTE FUNCTION public.lab_arena_restart_claim_gate_v1()')
        q.execute("INSERT INTO route645_trigger_control VALUES('pending','arena-2026-11-06',%s)",(OTHER,))
        q.execute("UPDATE route645_trigger_control SET status='leased'")
        assert prior.scalar(q,'SELECT status FROM route645_trigger_control')=='leased'
    p.conn.rollback()


def test_restart_control_fields_remain_usable_and_unmodified(prepared):
    p=prepared
    with p.conn.cursor()as q:before=prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
    route(p)
    with p.conn.cursor()as q:
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='fixture645' WHERE singleton")
    p.conn.commit()
    assert claim(p,PRIMARY)=={'status':'paused'}
    with p.conn.cursor()as q:
        after=prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
    for field in ('guard_generation','guard_commitment','owner_commitment','guard_expires_at','candidate_commit','restart_scope','restart_phase','captured_leases','actor_ref'):
        assert after[field]==before[field]


def test_exact_activation_preserves_every_run_round_bill_and_replays(prepared):
    p=prepared
    before=prior.snapshot(p)
    with p.conn.cursor()as q:
        control=prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
    sql=SQL.read_text().replace('5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9',PRIMARY).replace('2026-11-06T09:35:00Z','2099-11-06T09:35:00Z')
    prior.apply(p,sql)
    assert prior.snapshot(p)==before
    with p.conn.cursor()as q:
        after=prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
    assert {k:v for k,v in after.items()if k not in FIELDS}=={k:v for k,v in control.items()if k not in FIELDS}
    assert after['new_claim_round_id']==ROUND and after['new_claim_allowed_runner_hotkey']==PRIMARY
    prior.apply(p,sql)
    assert prior.snapshot(p)==before


sixth_prepared=native.prepared


def test_active_route_composes_with_exact_sixth_and_no_seventh(sixth_prepared):
    p=sixth_prepared
    route(p,runner=native.hotkey('633-runner'))
    with p.conn.cursor()as q:before=prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
    native.recovery.apply(p)
    with p.conn.cursor()as q:
        assert prior.scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')==before
        assert prior.scalar(q,"SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),'sha256'),'hex')",(native.CLAIM,))=='2acc6ff6c64ed797299cd18f203b3d62459720e2ae1ab9cb1f03a7850e62dd82'
    with p.conn.cursor()as q:
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s",(native.RUN,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    lease=native.claim(p)
    assert lease['status']=='leased'and lease['run_id']==native.RUN
    assert native.recovery.complete(p,lease,cause='judge_error')['status']=='failed'
    p.store.expire_leases(ROUND)
    with p.conn.cursor()as q:
        assert prior.scalar(q,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>6',(native.ASSIGNMENT,))==0


@pytest.mark.parametrize('fault',['acl','invoker'])
def test_migration_refuses_weakened_trigger_security(prepared,fault):
    p=prepared
    with p.conn.cursor()as q:
        if fault=='acl':q.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_restart_claim_gate_v1() TO lab_arena_service')
        else:q.execute('ALTER FUNCTION public.lab_arena_restart_claim_gate_v1() SECURITY INVOKER')
    p.conn.commit()
    before=prior.snapshot(p)
    try:
        with pytest.raises(p.pg.Error,match='Route645 trigger security differs'):
            prior.apply(p,SQL.read_text())
        p.conn.rollback()
        assert prior.snapshot(p)==before
    finally:
        with p.conn.cursor()as q:
            if fault=='acl':q.execute('REVOKE EXECUTE ON FUNCTION public.lab_arena_restart_claim_gate_v1() FROM lab_arena_service')
            else:q.execute('ALTER FUNCTION public.lab_arena_restart_claim_gate_v1() SECURITY DEFINER')
        p.conn.commit()


@pytest.mark.parametrize('fault',['disabled_trigger','detached_trigger','check_true','column_default'])
def test_migration_rejects_broken_schema_or_trigger_attachment(prepared,fault):
    p=prepared
    with p.conn.cursor()as q:
        trigger=prior.scalar(q,"SELECT pg_get_triggerdef(oid) FROM pg_trigger WHERE tgrelid='public.lab_arena_runs'::regclass AND tgname='lab_arena_runs_restart_claim_gate'")
        constraint=prior.scalar(q,"SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid='public.lab_arena_restart_claim_control'::regclass AND conname='lab_arena_new_claim_route_complete_v1'")
        if fault=='disabled_trigger':q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_restart_claim_gate')
        elif fault=='detached_trigger':q.execute('DROP TRIGGER lab_arena_runs_restart_claim_gate ON public.lab_arena_runs')
        elif fault=='check_true':
            q.execute('ALTER TABLE public.lab_arena_restart_claim_control DROP CONSTRAINT lab_arena_new_claim_route_complete_v1')
            q.execute('ALTER TABLE public.lab_arena_restart_claim_control ADD CONSTRAINT lab_arena_new_claim_route_complete_v1 CHECK(TRUE)')
        else:q.execute("ALTER TABLE public.lab_arena_restart_claim_control ALTER COLUMN new_claim_actor_ref SET DEFAULT 'unexpected'")
    p.conn.commit();before=prior.snapshot(p)
    try:
        with pytest.raises(p.pg.Error,match='Route645 (trigger attachment|constraint shape|column shape) differs'):
            prior.apply(p,SQL.read_text())
        p.conn.rollback();assert prior.snapshot(p)==before
    finally:
        with p.conn.cursor()as q:
            if fault=='disabled_trigger':q.execute('ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_restart_claim_gate')
            elif fault=='detached_trigger':q.execute(trigger)
            elif fault=='check_true':
                q.execute('ALTER TABLE public.lab_arena_restart_claim_control DROP CONSTRAINT lab_arena_new_claim_route_complete_v1')
                q.execute('ALTER TABLE public.lab_arena_restart_claim_control ADD CONSTRAINT lab_arena_new_claim_route_complete_v1 '+constraint)
            else:q.execute('ALTER TABLE public.lab_arena_restart_claim_control ALTER COLUMN new_claim_actor_ref DROP DEFAULT')
        p.conn.commit()
