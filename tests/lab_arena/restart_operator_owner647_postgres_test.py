"""Operator ownership survives the native canonical guard lifecycle."""
from pathlib import Path
import json

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.test_lab_arena_restart_claim_drain_postgres import (
    CANDIDATE, GUARD, NEW_CANDIDATE, NEW_GUARD, OWNER, _rpc,
)

SCRIPTS=Path(__file__).parents[2]/'scripts'
SQL=SCRIPTS/'647-lab-arena-preserve-operator-hold-owner.sql'
HOLD_ACTOR='independent-operator'
HOLD_REASON='independent-investigation'
ROUND='arena-2026-11-05'
SIGNATURES=(
 'public.lab_arena_acquire_restart_guard_v1(text,text,bigint,integer,text,text,text)',
 'public.lab_arena_retarget_restart_guard_v1(text,text,bigint,text,text,text,integer,text)',
 'public.lab_arena_abort_restart_guard_v1(text,text,bigint,text)',
 'public.lab_arena_release_restart_guard_v1(text,text,bigint,text)',
)


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def scalar(q,sql,args=()):
    q.execute(sql,args);return q.fetchone()[0]


def snapshot(c):
    with c.cursor()as q:
        return {sig:scalar(q,"SELECT jsonb_build_object('source',pg_get_functiondef(p.oid),'metadata',to_jsonb(p)-'prosrc') FROM pg_proc p WHERE p.oid=to_regprocedure(%s)",(sig,))for sig in SIGNATURES}


def control(c):
    with c.cursor()as q:return scalar(q,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')


def apply(c):
    with c.cursor()as q:q.execute(SQL.read_text())


def hold(c):
    with c.cursor()as q:
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,actor_ref=%s,pause_reason=%s WHERE singleton",(HOLD_ACTOR,HOLD_REASON))


def assert_hold(c):
    s=control(c)
    assert (s['operator_paused'],s['pause_reason'],s['actor_ref'])==(True,HOLD_REASON,HOLD_ACTOR),'operator owner changed'


def acquire(c,*,generation=0,actor='canonical-acquire'):
    return _rpc(c,'lab_arena_acquire_restart_guard_v1',GUARD,OWNER,generation,600,CANDIDATE,'gateway',actor)


def release(c,*,guard=GUARD,generation=1,actor='canonical-release'):
    _rpc(c,'lab_arena_authorize_restart_phase_v1',guard,OWNER,generation,'gateway_destructive')
    _rpc(c,'lab_arena_mark_restart_ready_v1',guard,OWNER,generation,'gateway_ready')
    return _rpc(c,'lab_arena_release_restart_guard_v1',guard,OWNER,generation,actor)


def check_cycle(c):
    hold(c);assert_hold(c)
    assert acquire(c)['guard_present']is True;assert_hold(c)
    assert acquire(c,generation=1,actor='canonical-renew')['guard_generation']==1;assert_hold(c)
    assert release(c)['guard_present']is False;assert_hold(c)


def test_regression_fails_on_exact_previous_source(database):
    pg,dsn=database
    with pg.connect(**dsn)as c:
        c.autocommit=True
        with pytest.raises(AssertionError,match='operator owner changed'):check_cycle(c)
        assert control(c)['actor_ref']=='canonical-acquire'


def test_held_acquire_renew_release_preserve_original_owner(database):
    pg,dsn=database
    with pg.connect(**dsn)as c:
        c.autocommit=True;apply(c);check_cycle(c)


def test_held_retarget_and_abort_preserve_original_owner(database):
    pg,dsn=database
    with pg.connect(**dsn)as c:
        c.autocommit=True;apply(c);hold(c);acquire(c);assert_hold(c)
        _rpc(c,'lab_arena_abort_restart_guard_v1',GUARD,OWNER,1,'canonical-abort');assert_hold(c)
        acquire(c,generation=1);assert_hold(c)
        _rpc(c,'lab_arena_authorize_restart_phase_v1',GUARD,OWNER,2,'gateway_destructive')
        s=_rpc(c,'lab_arena_retarget_restart_guard_v1',GUARD,OWNER,2,NEW_GUARD,NEW_CANDIDATE,'gateway',600,'canonical-retarget')
        assert s['guard_generation']==3 and s['candidate_commit']==NEW_CANDIDATE;assert_hold(c)
        _rpc(c,'lab_arena_mark_restart_ready_v1',NEW_GUARD,OWNER,3,'gateway_ready')
        _rpc(c,'lab_arena_release_restart_guard_v1',NEW_GUARD,OWNER,3,'canonical-final-release');assert_hold(c)


def test_original_owner_can_release_hold_after_restart(database):
    pg,dsn=database
    with pg.connect(**dsn)as c:
        c.autocommit=True
        with c.cursor()as q:
            # Install the committed hold RPC only; its phase includes judged.
            src=(SCRIPTS/'517-lab-arena-model-policy-supersession.sql').read_text()
            q.execute('GRANT CREATE ON SCHEMA public TO lab_arena_owner')
            q.execute(src[src.index('CREATE OR REPLACE FUNCTION public.lab_arena_scoring_operator_hold_v1('):src.index('CREATE OR REPLACE FUNCTION public.lab_arena_scoring_policy_history_v1(')])
            q.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
            cfg=base_round_configuration()
            q.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc) VALUES(%s,'stage2_judged',8,7,%s::jsonb)",(ROUND,json.dumps(cfg)))
        apply(c);check_cycle(c)
        with c.cursor()as q:
            cfg=scalar(q,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))
        result=_rpc(c,'lab_arena_scoring_operator_hold_v1',ROUND,json.dumps(cfg),8,7,1,HOLD_ACTOR,HOLD_REASON,False)
        assert result=={'status':'released','operator_paused':False,'guard_generation':1}


def test_unpaused_actor_updates_and_reason_cleanup_are_unchanged(database):
    pg,dsn=database
    with pg.connect(**dsn)as c:
        c.autocommit=True;apply(c)
        acquire(c);assert control(c)['actor_ref']=='canonical-acquire'
        acquire(c,generation=1,actor='canonical-renew');assert control(c)['actor_ref']=='canonical-renew'
        _rpc(c,'lab_arena_abort_restart_guard_v1',GUARD,OWNER,1,'canonical-abort')
        assert control(c)['actor_ref']=='canonical-abort'and control(c)['pause_reason']==''
        acquire(c,generation=1)
        _rpc(c,'lab_arena_authorize_restart_phase_v1',GUARD,OWNER,2,'gateway_destructive')
        _rpc(c,'lab_arena_retarget_restart_guard_v1',GUARD,OWNER,2,NEW_GUARD,NEW_CANDIDATE,'gateway',600,'canonical-retarget')
        assert control(c)['actor_ref']=='canonical-retarget'
        _rpc(c,'lab_arena_mark_restart_ready_v1',NEW_GUARD,OWNER,3,'gateway_ready')
        _rpc(c,'lab_arena_release_restart_guard_v1',NEW_GUARD,OWNER,3,'canonical-release')
        assert control(c)['actor_ref']=='canonical-release'and control(c)['pause_reason']==''


@pytest.mark.parametrize('action',['acquire','retarget','abort','release'])
def test_foreign_guard_owner_is_still_refused(database,action):
    pg,dsn=database
    with pg.connect(**dsn)as c:
        c.autocommit=True;apply(c);hold(c);acquire(c)
        if action in ('retarget','release'):_rpc(c,'lab_arena_authorize_restart_phase_v1',GUARD,OWNER,1,'gateway_destructive')
        if action=='release':_rpc(c,'lab_arena_mark_restart_ready_v1',GUARD,OWNER,1,'gateway_ready')
        before=control(c);foreign='lab_arena_restart_owner:'+'f'*64
        args={
         'acquire':(GUARD,foreign,1,600,CANDIDATE,'gateway','foreign'),
         'retarget':(GUARD,foreign,1,NEW_GUARD,NEW_CANDIDATE,'gateway',600,'foreign'),
         'abort':(GUARD,foreign,1,'foreign'),
         'release':(GUARD,foreign,1,'foreign'),
        }
        with pytest.raises(pg.Error):_rpc(c,'lab_arena_'+action+'_restart_guard_v1',*args[action])
        assert control(c)==before;assert_hold(c)


def test_exact_metadata_schema_acl_and_replay_are_preserved(database):
    pg,dsn=database
    with pg.connect(**dsn)as c:
        c.autocommit=True;before=snapshot(c);state=control(c)
        with c.cursor()as q:acl=scalar(q,"SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        apply(c);after=snapshot(c)
        assert control(c)==state
        for sig in SIGNATURES:
            assert after[sig]['metadata']==before[sig]['metadata']
            assert after[sig]['source']==before[sig]['source'].replace('actor_ref = left(p_actor_ref, 200)','actor_ref = CASE WHEN operator_paused THEN actor_ref ELSE left(p_actor_ref, 200) END')
        apply(c);assert snapshot(c)==after and control(c)==state
        with c.cursor()as q:assert scalar(q,"SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")==acl


@pytest.mark.parametrize('fault',['held','guarded','changed_source','changed_acl'])
def test_install_refuses_unsafe_control_or_unreviewed_source_without_partial_change(database,fault):
    pg,dsn=database
    with pg.connect(**dsn)as c:
        c.autocommit=True
        if fault=='held':hold(c)
        if fault=='guarded':acquire(c)
        with c.cursor()as q:
            if fault=='changed_acl':q.execute('GRANT EXECUTE ON FUNCTION '+SIGNATURES[-1]+' TO anon')
            if fault=='changed_source':
                src=scalar(q,'SELECT pg_get_functiondef(to_regprocedure(%s))',(SIGNATURES[-1],))
                q.execute(src.replace('DECLARE','DECLARE\n-- unreviewed change',1))
            acl=scalar(q,"SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        c.commit()
        before=snapshot(c);state=control(c)
        with pytest.raises(pg.Error,match='Guard647'):apply(c)
        with c.cursor()as q:q.execute('ROLLBACK')
        assert snapshot(c)==before and control(c)==state
        with c.cursor()as q:assert scalar(q,"SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")==acl


def test_non_superuser_upgrade_preserves_owner_grants_and_schema_acl(database):
    pg,dsn=database
    with pg.connect(**dsn)as c:
        c.autocommit=True
        with c.cursor()as q:
            q.execute('CREATE ROLE guard647_migrator NOLOGIN NOSUPERUSER INHERIT')
            q.execute('GRANT lab_arena_owner TO guard647_migrator')
            q.execute('GRANT USAGE,CREATE ON SCHEMA public TO guard647_migrator WITH GRANT OPTION')
            q.execute("SELECT rolsuper FROM pg_roles WHERE rolname='guard647_migrator'")
            assert q.fetchone()==(False,)
            acl=scalar(q,"SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert scalar(q,"SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")is False
            q.execute('SET ROLE guard647_migrator')
        before=snapshot(c);apply(c);after=snapshot(c)
        assert all(after[s]['metadata']==before[s]['metadata']for s in SIGNATURES)
        apply(c);assert snapshot(c)==after
        with c.cursor()as q:
            q.execute('RESET ROLE')
            assert scalar(q,"SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")==acl
            assert scalar(q,"SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")is False
