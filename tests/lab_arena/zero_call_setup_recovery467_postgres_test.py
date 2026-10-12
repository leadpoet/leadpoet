"""Recorded cold-image setup faults release only authenticated zero-call leases."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from unittest.mock import patch
import uuid

import pytest

from tests.lab_arena import owner_overdue_lease_expiry463_postgres_test as deployed
from tests.lab_arena import proxy_model_capacity401_postgres_test as historical
from tests.lab_arena import abandoned_host_recovery412_postgres_test as recovery

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/467-lab-arena-zero-call-setup-host-recovery.sql'
PREHASH = '11a47cbe7ba950bdc284342a2ede73cd08aba7b0f55e452131d3bf87c9a81c15'
CASES = json.loads((ROOT / 'tests/fixtures/arena_recovery46_setup_errors.json').read_text())


@pytest.fixture(scope='module')
def database():
    # Install465 after the historical421/450/463 upgrade chain, as production
    # did. Installing it before420 changes that exact historical preimage.
    migrations = tuple(x for x in historical.CURRENT_SERVICE_MIGRATIONS if not x.startswith('465-'))
    with patch.object(historical, 'CURRENT_SERVICE_MIGRATIONS', migrations):
        generator = deployed.database.__wrapped__()
        value = next(generator)
    try:
        deployed.migrated.__wrapped__(value)
        with value[0].connect(**value[1]) as conn, conn.cursor() as cur:
            cur.execute((ROOT / 'scripts/465-lab-arena-partial-company-judgments.sql').read_text())
        yield value
    finally:
        generator.close()


def _reset_control(conn):
    deployed.prior._reset(conn)


def _event(conn, lease, kind, content, *, token='a'):
    batch = [dict(event_id=str(uuid.uuid4()), kind=kind,
                  occurred_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(), content=content)]
    with conn.cursor() as cur:
        cur.execute('SET ROLE lab_arena_service')
        cur.execute('SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)',
                    (lease['run_id'], 'sha256:' + token * 64, json.dumps(batch)))
        result = cur.fetchone()[0]
        cur.execute('RESET ROLE')
    conn.commit()
    return result


def _seed(conn, *, kind='execute', case=CASES[0], started=True, error=True, sole=False):
    _reset_control(conn)
    recovery.prior._seed(conn, {'terminal_status': 'judge_error'}, 'judge_error', kind=kind)
    with conn.cursor() as cur:
        cur.execute('SET session_replication_role=replica')
        cur.execute('DELETE FROM public.lab_arena_runs WHERE attempt=2')
        cur.execute("UPDATE public.lab_arena_runs SET status='pending',terminal_cause=NULL,result_doc=NULL,runner_hotkey=NULL,lease_generation=0")
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{lease_ttl_seconds}','4500'::jsonb)")
        if sole:
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{runner_hotkeys}',%s::jsonb)", (json.dumps([recovery.prior.RUNNER_A]),))
        cur.execute('SET session_replication_role=origin')
    conn.commit()
    lease = recovery._claim(conn, recovery.prior.RUNNER_A, 'a')
    assert lease['status'] == 'leased'
    if started:
        assert _event(conn, lease, 'runtime.started', {'status': 'starting', 'lease_generation': lease['lease_generation']})['status'] == 'accepted'
    if error:
        content = {key: case[key] for key in ('status', 'failure_stage', 'error_class')}
        assert _event(conn, lease, 'runtime.error', content)['status'] == 'accepted'
    return lease


def _state(cur):
    cur.execute("SELECT p.oid::regprocedure::text,pg_get_functiondef(p.oid),owner.rolname,p.proacl::text,p.prosecdef,p.provolatile,p.proconfig FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner WHERE p.oid=ANY(%s::regprocedure[]) ORDER BY p.oid",
                (["public.lab_arena_expire_leases(text)", "public.lab_arena_restart_quiescence_v1(text,text,bigint)", "public.lab_arena__restart_drain_state_v1()", "public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)"],))
    return cur.fetchall()


@pytest.fixture(scope='module')
def migrated(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        # The exact captured production failure stays leased on the old RPC.
        lease = _seed(conn)
        assert recovery._expire(conn) == {'status': 'ok', 'expired': 0, 'retried': 0}
        with conn.cursor() as cur:
            before = _state(cur)
            assert recovery.prior398._hash(cur, recovery.prior398.EXPIRY) == PREHASH
            cur.execute(SQL.read_text())
            after = _state(cur)
            cur.execute(SQL.read_text())
            assert _state(cur) == after
            assert [row[2:] for row in before] == [row[2:] for row in after]
            old, new = ({row[0]: row[1] for row in state} for state in (before, after))
            expiry = 'lab_arena_expire_leases(text)'
            for name in old.keys() - {expiry}:
                assert old[name] == new[name]
            start = '  -- lab_arena_abandoned_host_recovery_v1:'
            end = '  -- The existing whole-round expiry must never touch foreign captured work.'
            assert old[expiry].split(start)[0] == new[expiry].split(start)[0]
            assert old[expiry].split(end)[1] == new[expiry].split(end)[1]
        conn.commit()
        assert recovery._expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 1}
    return True


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['run_id'].split(':')[-2])
def test_captured_zero_call_setup_fault_recovers_with_frozen_claim_and_retry(database, migrated, case):
    assert len(CASES) == 7
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, case=case, sole=True)
        assert recovery._expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 1}
        with conn.cursor() as cur:
            cur.execute('SELECT status,terminal_cause,terminal_doc,claim_response,result_doc,output_ref FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            status, cause, doc, claim, result, output = cur.fetchone()
            assert (status, cause, claim, result, output) == ('failed', 'lease_expired', lease, None, None)
            assert doc['recovery_reason'] == 'authenticated_zero_call_setup_host_error'
            assert datetime.fromisoformat(doc['original_lease_expires_at']) > datetime.now(timezone.utc)
            cur.execute('SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone()[0] == 0
        # Once host capacity is fixed, the existing sole-runner fallback leases
        # the ordinary second execute attempt. No admission policy is changed.
        retry = recovery._claim(conn, recovery.prior.RUNNER_A, 'b')
        assert retry['status'] == 'leased' and retry['attempt'] == 2
        assert _event(conn, retry, 'runtime.started', {'status': 'starting', 'lease_generation': retry['lease_generation']}, token='b')['status'] == 'accepted'
        assert _event(conn, retry, 'runtime.error', {'status': 'abandoned', 'failure_stage': 'setup', 'error_class': 'RunnerError'}, token='b')['status'] == 'accepted'
        assert recovery._expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 0}
        assert recovery._claim(conn, recovery.prior.RUNNER_A, 'c') == {'status': 'no_pending'}


@pytest.mark.parametrize('kind', ['execute', 'score'])
def test_setup_recovery_uses_existing_cross_runner_handoff(database, migrated, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, kind=kind)
        assert recovery._expire(conn)['retried'] == 1
        retry = recovery._claim(conn, recovery.prior.RUNNER_B, 'b')
        assert retry['status'] == 'leased' and retry['attempt'] == 2
        assert retry['run_id'] == lease['assignment_id'] + ':2'


@pytest.mark.parametrize('fault', [
    'live', 'nostart', 'wrong_start_generation', 'wrong_claim_generation', 'wrong_start_runner',
    'wrong_runner', 'wrong_round', 'wrong_submission', 'wrong_miner', 'wrong_assignment',
    'wrong_stage', 'wrong_position', 'wrong_attempt', 'wrong_kind', 'wrong_class', 'model_error',
    'runtime_stage', 'cleanup_stage', 'wrong_status', 'ledger', 'paid', 'provider', 'output', 'result',
    'cleanup_event', 'finished', 'before_claim', 'after_expiry', 'old_generation', 'other_stage',
])
def test_setup_recovery_keeps_every_lease_provider_and_cleanup_fence(database, migrated, fault):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, started=fault != 'nostart', error=fault != 'live')
        with conn.cursor() as cur:
            cur.execute('SET session_replication_role=replica')
            if fault in ('ledger', 'paid'):
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) VALUES (%s,%s,%s,%s,%s,1,%s,'deepline','exa.contents','host',%s)", ('settlement' if fault == 'paid' else 'reservation', recovery.prior.MINER, recovery.prior.ROUND, recovery.prior.SUBMISSION, lease['run_id'], 'sha256:' + 'f' * 64, 100 if fault == 'paid' else 0))
            elif fault in ('output', 'result', 'old_generation', 'other_stage', 'wrong_claim_generation'):
                mutation = {'output': "output_ref='present'", 'result': "result_doc='{}'::jsonb", 'old_generation': 'stage_generation=0', 'other_stage': 'stage=2', 'wrong_claim_generation': "claim_response=jsonb_set(claim_response,'{lease_generation}','99'::jsonb)"}[fault]
                cur.execute('UPDATE public.lab_arena_runs SET ' + mutation + ' WHERE run_id=%s', (lease['run_id'],))
            elif fault in ('cleanup_event', 'provider', 'finished'):
                kind = {'cleanup_event': 'runtime.cleanup_error', 'provider': 'provider.request', 'finished': 'runtime.finished'}[fault]
                cur.execute("UPDATE public.lab_arena_trajectory_events SET event_kind=%s WHERE event_kind='runtime.error'", (kind,))
            elif fault in ('wrong_class', 'model_error', 'runtime_stage', 'cleanup_stage', 'wrong_status'):
                key, value = {'wrong_class': ('error_class', 'OSError'), 'model_error': ('error_class', 'ModelError'), 'runtime_stage': ('failure_stage', 'runtime'), 'cleanup_stage': ('failure_stage', 'cleanup'), 'wrong_status': ('status', 'accepted')}[fault]
                cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,%s::text[],to_jsonb(%s::text)) WHERE event_kind='runtime.error'", ([key], value))
            elif fault == 'wrong_start_generation':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,'{lease_generation}','99'::jsonb) WHERE event_kind='runtime.started'")
            elif fault == 'wrong_start_runner':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET runner_hotkey=%s WHERE event_kind='runtime.started'", (recovery.prior.RUNNER_B,))
            elif fault.startswith('wrong_'):
                column, value = {'wrong_runner': ('runner_hotkey', recovery.prior.RUNNER_B), 'wrong_round': ('round_id', 'other'), 'wrong_submission': ('submission_id', 'other'), 'wrong_miner': ('miner_hotkey', recovery.prior.RUNNER_B), 'wrong_assignment': ('assignment_id', 'other'), 'wrong_stage': ('stage', 2), 'wrong_position': ('icp_position', 9), 'wrong_attempt': ('attempt', 2), 'wrong_kind': ('run_kind', 'score')}[fault]
                cur.execute('UPDATE public.lab_arena_trajectory_events SET ' + column + "=%s WHERE event_kind='runtime.error'", (value,))
            elif fault == 'before_claim':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET occurred_at=now()-interval '1 day' WHERE event_kind='runtime.error'")
            elif fault == 'after_expiry':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET created_at=now()+interval '1 day' WHERE event_kind='runtime.error'")
            cur.execute('SET session_replication_role=origin')
        conn.commit()
        assert recovery._expire(conn) == {'status': 'ok', 'expired': 0, 'retried': 0}


@pytest.mark.parametrize('hold', ['operator', 'guard'])
def test_existing_pause_and_guard_hold_early_setup_recovery(database, migrated, hold):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn)
        if hold == 'guard':
            generation = deployed.prior.prior._hold_capture(conn)
            before = deployed.guard._quiescence(conn, generation=generation)
            assert before['still_leased_count'] == 1 and before['expired_receipt_count'] == 0
        else:
            with conn.cursor() as cur:
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton")
            conn.commit()
        assert recovery._expire(conn)['expired'] == 0
        with conn.cursor() as cur:
            cur.execute('SELECT status,claim_response FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone() == ('leased', lease)


@pytest.mark.parametrize('winner', ['provider', 'recovery'])
def test_setup_recovery_and_provider_reservation_keep_existing_lock_fence(database, migrated, winner):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn)
    owner = psycopg.connect(**dsn)
    try:
        if winner == 'provider':
            assert recovery._reserve(owner, lease)['status'] == 'reserved'
        else:
            with owner.cursor() as cur:
                cur.execute('SELECT public.lab_arena_expire_leases(%s)', (recovery.prior.ROUND,))
                assert cur.fetchone()[0]['expired'] == 1
        started = threading.Event()
        def contender():
            with psycopg.connect(**dsn) as other:
                started.set()
                return recovery._expire(other) if winner == 'provider' else recovery._reserve(other, lease)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(contender)
            assert started.wait(2)
            with psycopg.connect(**dsn) as observer:
                for _ in range(100):
                    with observer.cursor() as cur:
                        cur.execute("SELECT count(*) FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE 'SELECT public.lab_arena_%'")
                        if cur.fetchone()[0]:
                            break
                    threading.Event().wait(.01)
                else:
                    pytest.fail('provider/recovery race did not reach the database lock')
            owner.commit()
            result = future.result(timeout=5)
        assert result['expired'] == 0 if winner == 'provider' else result['status'] == 'stale'
    finally:
        owner.close()


def test_setup_error_from_rejected_lease_cannot_authorize_recovery(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, error=False)
        assert _event(conn, lease, 'runtime.error', {'status': 'abandoned', 'failure_stage': 'setup', 'error_class': 'RunnerError'}, token='f')['status'] == 'stale'
        assert recovery._expire(conn)['expired'] == 0


def test_prior_runtime_host_recovery_reason_is_unchanged(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, error=False)
        assert _event(conn, lease, 'runtime.error', {'status': 'abandoned', 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError'})['status'] == 'accepted'
        assert recovery._expire(conn)['retried'] == 1
        with conn.cursor() as cur:
            cur.execute('SELECT terminal_doc FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone()[0]['recovery_reason'] == 'authenticated_zero_call_runtime_host_error'


def test_migration_security_and_preimage_drift_remain_fail_closed(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        with conn.cursor() as cur:
            before = _state(cur)
            cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_expire_leases(text) TO anon')
            with pytest.raises(psycopg.Error, match='security shape differs'):
                cur.execute(SQL.read_text())
        conn.rollback()
        with conn.cursor() as cur:
            assert _state(cur) == before
            cur.execute("SELECT pg_get_functiondef('public.lab_arena_expire_leases(text)'::regprocedure)")
            source = cur.fetchone()[0]
            cur.execute(source.replace('-- lab_arena_zero_call_setup_host_recovery_v1:', '-- unrelated source drift:'))
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(SQL.read_text())
        conn.rollback()
        with conn.cursor() as cur:
            assert _state(cur) == before
