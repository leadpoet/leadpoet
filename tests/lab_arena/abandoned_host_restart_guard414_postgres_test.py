"""Restart capture and operator pause retain natural-expiry receipt semantics."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import threading

import pytest

from tests.lab_arena import abandoned_host_recovery412_postgres_test as recovery
from tests.lab_arena import restart_expired_zero_call394_postgres_test as guard
from tests.lab_arena import score_host_retry398_postgres_test as identity

ROOT = Path(__file__).parents[2]
SQL = ROOT/'scripts/414-lab-arena-abandoned-host-restart-guard.sql'
PREHASH = 'ac1e99fd5bc1b4a458602ce05012d562847bb37d3cf4e69857e45db19077fd00'
database = recovery.database


def _generation(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control')
        return cur.fetchone()[0]


def _abort(conn, generation):
    guard._rpc(conn,'lab_arena_abort_restart_guard_v1',guard.GUARD,guard.OWNER,generation,'test-abort')
    conn.commit()


def _mixed(conn, kind='execute'):
    fault=recovery._seed(conn,kind=kind)
    with conn.cursor() as cur:
        cur.execute('SET session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET status='leased',runner_hotkey=%s,lease_token_hash=%s,lease_generation=1,lease_expires_at=now()-interval '1 second' WHERE run_id<>%s",(recovery.prior.RUNNER_B,'sha256:'+'b'*64,fault['run_id']))
        cur.execute('SET session_replication_role=origin')
    conn.commit()
    return fault


@pytest.fixture(scope='module')
def migrated(database):
    recovery.migrated.__wrapped__(database)
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            cur.execute((ROOT/'scripts/404-lab-arena-restart-closed-ledger-expiry.sql').read_text())
        _mixed(conn)
        generation=guard._acquire(conn,_generation(conn))['guard_generation']
        before_bug=guard._quiescence(conn,generation=generation)
        assert before_bug['lost_or_mutated_count']==1 and before_bug['preserved'] is False
        _abort(conn,generation)
        with conn.cursor() as cur:
            before=identity._security(cur)
            assert identity._hash(cur,identity.EXPIRY)==PREHASH
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error,match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREHASH,'0'*64,1))
            cur.execute('ROLLBACK')
            cur.execute(SQL.read_text())
            post=identity._hash(cur,identity.EXPIRY)
            cur.execute(SQL.read_text())
            assert identity._hash(cur,identity.EXPIRY)==post
            assert identity._security(cur)==before
    return True


@pytest.mark.parametrize('kind',['score','execute'])
def test_guard_preserves_unexpired_fault_then_natural_expiry_receipts(database,migrated,kind):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        fault=_mixed(conn,kind)
        generation=guard._acquire(conn,_generation(conn))['guard_generation']
        drained=guard._quiescence(conn,generation=generation)
        assert (drained['captured_count'],drained['expired_receipt_count'],drained['still_leased_count'],drained['lost_or_mutated_count'])==(2,1,1,0)
        with conn.cursor() as cur:
            cur.execute('SELECT status,terminal_doc,lease_expires_at FROM public.lab_arena_runs WHERE run_id=%s',(fault['run_id'],))
            state,doc,expiry=cur.fetchone()
            assert state=='leased' and doc is None and expiry>datetime.now(timezone.utc)
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s",(fault['run_id'],))
        finished=guard._quiescence(conn,generation=generation)
        assert finished['preserved'] is True
        assert (finished['expired_receipt_count'],finished['lost_or_mutated_count'],finished['pending_retry_count'])==(2,0,2)
        assert guard._quiescence(conn,generation=generation)['outcome_commitment']==finished['outcome_commitment']
        with conn.cursor() as cur:
            cur.execute('SELECT terminal_doc FROM public.lab_arena_runs WHERE run_id=%s',(fault['run_id'],))
            assert set(cur.fetchone()[0])=={'expired_at'}
        _abort(conn,generation)


@pytest.mark.parametrize('kind',['score','execute'])
def test_guard_abort_reenables_same_early_recovery(database,migrated,kind):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        fault=recovery._seed(conn,kind=kind)
        generation=guard._acquire(conn,_generation(conn))['guard_generation']
        conn.commit()
        assert recovery._expire(conn)['expired']==0
        _abort(conn,generation)
        assert recovery._expire(conn)=={'status':'ok','expired':1,'retried':1}
        assert recovery._expire(conn)['expired']==0
        assert recovery._claim(conn,recovery.prior.RUNNER_B,'d')['status']=='leased'


@pytest.mark.parametrize('hold',['operator','expired_guard'])
def test_unreleased_hold_disables_only_early_recovery(database,migrated,hold):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        fault=_mixed(conn)
        if hold=='operator':
            with conn.cursor() as cur:
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE,pause_reason='operator' WHERE singleton")
            conn.commit()
        else:
            generation=guard._acquire(conn,_generation(conn))['guard_generation']
            with conn.cursor() as cur:
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET guard_expires_at=now()-interval '1 second' WHERE singleton")
            conn.commit()
        assert recovery._expire(conn)=={'status':'ok','expired':1,'retried':1}
        with conn.cursor() as cur:
            cur.execute('SELECT status,terminal_doc FROM public.lab_arena_runs WHERE run_id=%s',(fault['run_id'],))
            assert cur.fetchone()==('leased',None)
        if hold=='operator':
            with conn.cursor() as cur:
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason='' WHERE singleton")
            conn.commit()
        else: _abort(conn,generation)
        assert recovery._expire(conn)['expired']==1


@pytest.mark.parametrize('winner',['capture','recovery'])
def test_capture_and_early_recovery_share_claim_control_lock(database,migrated,winner):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        fault=recovery._seed(conn)
        generation=_generation(conn)
    owner=psycopg.connect(**dsn)
    try:
        if winner=='capture':
            acquired=guard._acquire(owner,generation)
            generation=acquired['guard_generation']
            assert acquired['drain']['captured_count']==1
        else:
            with owner.cursor() as cur:
                cur.execute('SELECT public.lab_arena_expire_leases(%s)',(recovery.prior.ROUND,))
                assert cur.fetchone()[0]['expired']==1
        started=threading.Event()
        def losing_call():
            with psycopg.connect(**dsn) as loser:
                started.set()
                return recovery._expire(loser) if winner=='capture' else guard._acquire(loser,generation)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(losing_call)
            assert started.wait(2)
            with psycopg.connect(**dsn) as observer:
                for _ in range(100):
                    with observer.cursor() as cur:
                        cur.execute("SELECT count(*) FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE 'SELECT public.lab_arena_%'")
                        if cur.fetchone()[0]: break
                    threading.Event().wait(.01)
                else: pytest.fail('capture/recovery race never reached DB lock')
            owner.commit()
            answer=future.result(timeout=5)
        if winner=='capture':
            assert answer['expired']==0
            with owner.cursor() as cur:
                cur.execute('SELECT status FROM public.lab_arena_runs WHERE run_id=%s',(fault['run_id'],))
                assert cur.fetchone()[0]=='leased'
        else:
            generation=answer['guard_generation']
            assert answer['drain']['captured_count']==0
            assert answer['drain']['lost_or_mutated_count']==0
        _abort(owner,generation)
    finally: owner.close()


def test_http_host_recovery_publishes_and_promotes_with414_active(database,migrated,tmp_path,monkeypatch):
    from tests.lab_arena import abandoned_host_recovery412_e2e_test as full_http
    full_http.test_host_fault_recovers_via_signed_http_run_then_publishes_and_promotes(
        database,migrated,tmp_path,monkeypatch,
    )


def test_guarded_closed_paid_receipts_are_unchanged_with414_active(database,migrated):
    from tests.lab_arena import restart_closed_ledger404_postgres_test as paid
    paid.test_paid_closed_heads_expire_without_receipt_change(database)


@pytest.mark.parametrize('head',['reservation','dispatch'])
def test_guarded_open_provider_work_stays_leased_with414_active(database,migrated,head):
    from tests.lab_arena import restart_closed_ledger404_postgres_test as paid
    paid.test_open_head_stays_leased(database,head)


def test_guarded_unidentified_cost_stays_leased_with414_active(database,migrated):
    from tests.lab_arena import restart_closed_ledger404_postgres_test as paid
    paid.test_unidentified_ledger_cannot_prove_closed(database)
