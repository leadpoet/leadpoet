"""Owner-authorized canonical drain reuses stopped-host proof without new claims."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import completed_score_host_recovery450_postgres_test as prior
from tests.lab_arena import restart_expired_zero_call394_postgres_test as guard

ROOT = Path(__file__).parents[2]
SQL = ROOT/'scripts/454-lab-arena-owner-authorized-stopped-score-drain.sql'
database = prior.database
CONTEXT = 'lab_arena.restart_recovery_owner'
SIGNATURES = guard.FUNCTIONS + ('public.lab_arena_expire_leases(text)',)


def _state(cur):
    cur.execute('SELECT p.oid::regprocedure::text,pg_get_functiondef(p.oid),owner.rolname,'
                'p.proacl::text,p.prosecdef,p.provolatile,p.proconfig FROM pg_proc p '
                'JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=ANY(%s::regprocedure[]) ORDER BY p.oid',
                (list(SIGNATURES),))
    return cur.fetchall()


def _hold_capture(conn):
    with conn.cursor() as cur:
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE,"
                    "pause_reason='oct10_uniform_search_date_recovery',actor_ref='fixture451' WHERE singleton")
        cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control')
        generation = cur.fetchone()[0]
    conn.commit()
    generation = guard._acquire(conn,generation)['guard_generation']
    conn.commit()
    return generation


@pytest.fixture(scope='module')
def migrated(database):
    prior.migrated.__wrapped__(database)
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        # Actual pre-fix defect: authenticated stopped settled scorer cannot
        # drain under a held canonical capture before its frozen TTL.
        lease=prior.prior._seed(conn)
        generation=_hold_capture(conn)
        before=guard._quiescence(conn,generation=generation)
        assert (before['still_leased_count'],before['expired_receipt_count'],before['preserved'])==(1,0,False)
        with conn.cursor() as cur:
            state=_state(cur)
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error,match='preimage differs'):
                cur.execute(SQL.read_text().replace('1539836b593cee82a8e9ac237199208464da5aa5bd960e9179275c9f51387f97','0'*64,1))
            cur.execute('ROLLBACK')
            assert _state(cur)==state
            cur.execute(SQL.read_text())
            after=_state(cur)
            assert [row[2:] for row in after]==[row[2:] for row in state]
            old={row[0]:row[1] for row in state}
            new={row[0]:row[1] for row in after}
            expiry='lab_arena_expire_leases(text)'
            tail="  FOR v_run IN\n    SELECT * FROM public.lab_arena_runs\n    WHERE round_id = p_round_id AND status = 'leased'"
            assert old[expiry].split(tail)[1]==new[expiry].split(tail)[1]
            assert new[expiry].count('lab_arena_owner_authorized_stopped_score_drain_v1')==1
            zero='  -- lab_arena_abandoned_host_recovery_v1:'
            paid='  -- lab_arena_settled_execute_host_recovery_v1:'
            boundary='  -- The existing whole-round expiry must never touch foreign captured work.'
            assert old[expiry].split(zero)[1].split(paid)[0]==new[expiry].split(zero)[1].split(boundary)[0]
            assert new['lab_arena__restart_drain_state_v1()'].count('lab_arena__deepline_cost_binding_v1')==1
            cur.execute(SQL.read_text())
            assert _state(cur)==after
        fixed=guard._quiescence(conn,generation=generation)
        assert (fixed['expired_receipt_count'],fixed['preserved'])==(1,True)
    return True


def _context(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT current_setting(%s,TRUE)',(CONTEXT,))
        return cur.fetchone()[0]


@pytest.fixture(autouse=True)
def fresh_control(database,migrated):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn,conn.cursor() as cur:
        # Disposable test state only; each case must acquire its own real guard.
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,"
                    "pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',"
                    "guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',"
                    "captured_leases='[]'")


@pytest.mark.parametrize('count',[0,1])
def test_held_owner_drain_preserves_costs_frozen_claim_and_authorizes(database,migrated,count):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        lease,calls=prior._seed(conn,count=count)
        generation=_hold_capture(conn)
        before=prior.prior._audit(conn,lease)
        cache=prior._cache(conn,lease)
        # Ordinary expiry still cannot recover a held captured lease.
        assert prior.prior._expire(conn)['expired']==0
        with pytest.raises(psycopg.Error,match='drain_not_preserved'):
            guard._rpc(conn,'lab_arena_authorize_restart_phase_v1',guard.GUARD,guard.OWNER,generation,'gateway_destructive')
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute('SELECT set_config(%s,%s,FALSE)',(CONTEXT,'{"previous":"context"}'))
        conn.commit()
        drained=guard._quiescence(conn,generation=generation)
        assert _context(conn)=='{"previous":"context"}'
        assert (drained['captured_count'],drained['expired_receipt_count'],drained['lost_or_mutated_count'],drained['preserved'])==(1,1,0,True)
        assert guard._quiescence(conn,generation=generation)['outcome_commitment']==drained['outcome_commitment']
        after=prior.prior._audit(conn,lease)
        assert after['lab_arena_ledger']==before['lab_arena_ledger']
        assert after['lab_arena_trajectory_events']==before['lab_arena_trajectory_events']
        assert prior._cache(conn,lease)==cache
        assert after['run'][0:2]==('failed','lease_expired') and after['run'][3]==lease
        assert after['run'][2]['original_lease_expires_at']==before['run'][4].isoformat()
        assert drained['pending_retry_count']==1
        assert prior.prior.recovery._claim(conn,prior.prior.recovery.prior.RUNNER_B,'b')['status']=='paused'
        authorized=guard._rpc(conn,'lab_arena_authorize_restart_phase_v1',guard.GUARD,guard.OWNER,generation,'gateway_destructive')
        assert authorized['restart_phase']=='gateway_destructive'
        with conn.cursor() as cur:
            cur.execute('SELECT operator_paused FROM public.lab_arena_restart_claim_control')
            assert cur.fetchone()[0] is True


@pytest.mark.parametrize('fault',['owner','generation','context','uncaptured','capture_generation','wrong_runner','frozen_expiry','nohost','cleanup','dispatch','overdue_dispatch','failed_unknown','execute'])
def test_recovery_requires_exact_owner_capture_and_stopped_settled_proof(database,migrated,fault):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        lease=prior.prior._seed(conn,abandoned=fault!='nohost',settled=fault not in ('dispatch','overdue_dispatch'),kind='execute' if fault=='execute' else 'score')
        generation=_hold_capture(conn)
        before=prior.prior._audit(conn,lease)
        with conn.cursor() as cur:
            if fault=='uncaptured':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases='[]'")
            elif fault=='capture_generation':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases=jsonb_set(captured_leases,'{0,lease_generation}','999')")
            elif fault=='cleanup':
                prior.prior._event(conn,lease,'runtime.cleanup_error',{})
            elif fault in ('wrong_runner','frozen_expiry','overdue_dispatch'):
                cur.execute('SET session_replication_role=replica')
                if fault=='wrong_runner':
                    cur.execute('UPDATE public.lab_arena_runs SET runner_hotkey=%s WHERE run_id=%s',
                                (prior.prior.recovery.prior.RUNNER_B,lease['run_id']))
                elif fault=='frozen_expiry':
                    cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb((now()+interval '1 day')::text)) WHERE run_id=%s",(lease['run_id'],))
                else:
                    cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s",(lease['run_id'],))
                cur.execute('SET session_replication_role=origin')
            elif fault=='failed_unknown':
                assert prior.prior._reserve(conn,lease,prior.prior.SECOND_CALL)['status']=='reserved'
                assert prior.prior._dispatch(conn,lease,prior.prior.SECOND_CALL)['status']=='dispatched'
                assert prior.prior._rpc(conn,'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
                    (lease['run_id'],prior.prior.TOKEN_HASH,prior.prior.SECOND_CALL,json.dumps({'call_succeeded':False,'provider_status':500})))['status']=='uncertain'
        conn.commit()
        audit=prior.prior._audit(conn,lease)
        if fault in ('owner','generation'):
            with pytest.raises(psycopg.Error,match='owner_or_generation_differs'):
                guard._quiescence(conn,owner='lab_arena_restart_owner:'+'f'*64 if fault=='owner' else guard.OWNER,
                                  generation=generation+1 if fault=='generation' else generation)
            conn.rollback()
        elif fault=='context':
            with conn.cursor() as cur:
                cur.execute('SELECT set_config(%s,%s,TRUE)',(CONTEXT,json.dumps({'guard_id':guard.GUARD,'owner_id':'wrong','guard_generation':generation})))
            assert prior.prior._expire(conn)['expired']==0
        else:
            drained=guard._quiescence(conn,generation=generation)
            assert drained['preserved'] is False and drained['expired_receipt_count']==0
        assert prior.prior._audit(conn,lease)==audit
        with conn.cursor() as cur:
            cur.execute('SELECT status FROM public.lab_arena_runs WHERE run_id=%s',(lease['run_id'],))
            assert cur.fetchone()[0]=='leased'


def test_foreign_live_lease_rejects_owned_round_write_set_and_restores_context(database,migrated):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        lease=prior.prior._seed(conn)
        generation=_hold_capture(conn)
        with conn.cursor() as cur:
            cur.execute('SET session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,lease_token_hash=%s,lease_generation=1,lease_expires_at=now()+interval '1 hour' WHERE run_id<>%s",
                        (prior.prior.recovery.prior.RUNNER_B,'sha256:'+'b'*64,lease['run_id']))
            cur.execute('SET session_replication_role=origin')
            cur.execute('SELECT set_config(%s,%s,FALSE)',(CONTEXT,'{"previous":"context"}'))
        conn.commit()
        before=prior.prior._audit(conn,lease)
        with pytest.raises(psycopg.Error,match='captured write set differs'):
            guard._quiescence(conn,generation=generation)
        conn.rollback()
        assert _context(conn)=='{"previous":"context"}'
        assert prior.prior._audit(conn,lease)==before
