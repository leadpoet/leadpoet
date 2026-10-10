"""Paid score abandonment uses actual current PostgreSQL lease/provider RPCs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

import pytest

from tests.lab_arena import abandoned_host_recovery412_postgres_test as recovery
from tests.lab_arena import abandoned_host_restart_guard414_postgres_test as restart
from tests.lab_arena import oct07_cutoff_recovery419_postgres_test as current
from tests.lab_arena import score_host_retry398_postgres_test as identity

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/441-lab-arena-settled-score-host-recovery.sql'
PREHASH = 'ef61f7bf17884267c7c126529148099e48591eb973e88ecb841eea4498439688'
database = recovery.database
TOKEN_HASH = 'sha256:' + 'a' * 64
CALL = 'sha256:' + 'e' * 64
SECOND_CALL = 'sha256:' + 'f' * 64


@pytest.fixture(scope='module')
def migrated(database):
    # Install the production upgrade chain, not a handwritten expiry function.
    current.database.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for number in (342, 377, *range(420, 433), *range(434, 440)):
                paths = list((ROOT / 'scripts').glob(f'{number}-*.sql'))
                assert len(paths) == 1
                cur.execute(paths[0].read_text())
            before = identity._security(cur)
            assert identity._hash(cur, identity.EXPIRY) == PREHASH
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREHASH, '0' * 64, 1))
            cur.execute('ROLLBACK')
            cur.execute(SQL.read_text())
            post = identity._hash(cur, identity.EXPIRY)
            cur.execute(SQL.read_text())
            assert identity._hash(cur, identity.EXPIRY) == post
            assert identity._security(cur) == before
    return True


def _rpc(conn, expression, args=(), *, commit=True):
    with conn.cursor() as cur:
        cur.execute('SET ROLE lab_arena_service')
        cur.execute('SELECT public.' + expression, args)
        value = cur.fetchone()[0]
        cur.execute('RESET ROLE')
    if commit:
        conn.commit()
    return value


def _event(conn, lease, kind, content, *, token_hash=TOKEN_HASH):
    batch = [{'event_id': str(uuid.uuid4()), 'kind': kind,
              'occurred_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
              'content': content}]
    assert _rpc(conn, 'lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)',
                (lease['run_id'], token_hash, json.dumps(batch)))['status'] == 'accepted'


def _reserve(conn, lease, call=CALL, *, commit=True, token_hash=TOKEN_HASH):
    return _rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'exa.contents','deepline','host',0,%s::jsonb,4500)",
                (lease['run_id'], token_hash, call,
                 json.dumps({'reserve_remaining_budget': True})), commit=commit)


def _dispatch(conn, lease, call=CALL, *, commit=True, token_hash=TOKEN_HASH):
    return _rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
                (lease['run_id'], token_hash, call), commit=commit)


def _settle(conn, lease, call=CALL, *, commit=True, token_hash=TOKEN_HASH):
    return _rpc(conn, "lab_arena_settle_call(%s,%s,%s,100,%s::jsonb,4500)",
                (lease['run_id'], token_hash, call,
                 json.dumps({'status': 200, 'call': {'call_succeeded': True}})), commit=commit)


def _expire(conn, *, commit=True):
    return _rpc(conn, 'lab_arena_expire_leases(%s)', (recovery.prior.ROUND,), commit=commit)


def _seed(conn, *, abandoned=True, kind='score', settled=True):
    lease = recovery._seed(conn, kind=kind, event=False)
    _event(conn, lease, 'runtime.started', {'status': 'starting', 'runtime': 'runsc',
                                         'lease_generation': lease['lease_generation']})
    assert _reserve(conn, lease)['status'] == 'reserved'
    assert _dispatch(conn, lease)['status'] == 'dispatched'
    if settled:
        assert _settle(conn, lease)['status'] == 'settled'
    if abandoned:
        _event(conn, lease, 'runtime.error', {
            'status': 'abandoned', 'failure_stage': 'runtime',
            'error_class': 'RuntimeHostError',
            'runtime_host_reason': 'sandbox_launcher_signaled',
            'launch_exit_code': -2, 'launch_timed_out': False,
        })
    return lease


def _audit(conn, lease):
    result = {}
    with conn.cursor() as cur:
        for table, order in (('lab_arena_ledger', 'entry_id'),
                             ('lab_arena_trajectory_events', 'trajectory_id')):
            cur.execute(f'SELECT coalesce(jsonb_agg(to_jsonb(x) ORDER BY {order}),\'[]\') '
                        f'FROM public.{table} x WHERE run_id=%s', (lease['run_id'],))
            result[table] = cur.fetchone()[0]
        cur.execute('SELECT status,terminal_cause,terminal_doc,claim_response,lease_expires_at '
                    'FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
        result['run'] = cur.fetchone()
    return result


def test_last_settlement_after_abandonment_releases_once_with_renewed_lease(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, settled=False)
        assert recovery._claim(conn, recovery.prior.RUNNER_B, 'b') == {'status': 'no_pending'}
        assert _expire(conn) == {'status': 'ok', 'expired': 0, 'retried': 0}
        assert _settle(conn, lease)['status'] == 'settled'
        before = _audit(conn, lease)
        assert before['run'][4] > datetime.fromisoformat(lease['lease_expires_at'])
        assert _expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 1}
        assert _expire(conn) == {'status': 'ok', 'expired': 0, 'retried': 0}
        after = _audit(conn, lease)
        assert after['lab_arena_ledger'] == before['lab_arena_ledger']
        assert after['lab_arena_trajectory_events'] == before['lab_arena_trajectory_events']
        status, cause, doc, claim, expiry = after['run']
        assert (status, cause) == ('failed', 'lease_expired')
        assert doc['recovery_reason'] == 'authenticated_settled_score_runtime_host_error'
        assert datetime.fromisoformat(doc['original_lease_expires_at']) == before['run'][4]
        assert claim == lease and expiry <= datetime.now(timezone.utc)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('DELETE FROM public.lab_arena_runs WHERE assignment_id<>%s',
                        (lease['assignment_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        retry = recovery._claim(conn, recovery.prior.RUNNER_B, 'c')
        assert (retry['status'], retry['attempt']) == ('leased', 2)
        # A paid attempt 2 can close early but must never get attempt 3.
        retry_token = 'sha256:' + 'c' * 64
        _event(conn, retry, 'runtime.started', {'lease_generation': retry['lease_generation']}, token_hash=retry_token)
        assert _reserve(conn, retry, SECOND_CALL, token_hash=retry_token)['status'] == 'reserved'
        assert _dispatch(conn, retry, SECOND_CALL, token_hash=retry_token)['status'] == 'dispatched'
        assert _settle(conn, retry, SECOND_CALL, token_hash=retry_token)['status'] == 'settled'
        _event(conn, retry, 'runtime.error', {
            'status': 'abandoned', 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError',
            'runtime_host_reason': 'sandbox_launcher_signaled', 'launch_exit_code': -2,
            'launch_timed_out': False}, token_hash=retry_token)
        retry_audit = _audit(conn, retry)
        assert _expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 0}
        assert _audit(conn, retry)['lab_arena_ledger'] == retry_audit['lab_arena_ledger']
        with conn.cursor() as cur:
            cur.execute('SELECT max(attempt) FROM public.lab_arena_runs WHERE assignment_id=%s',
                        (lease['assignment_id'],))
            assert cur.fetchone()[0] == 2


@pytest.mark.parametrize('head', ['reservation', 'dispatch', 'uncertain', 'recovery', 'refusal'])
def test_any_nonsettlement_head_blocks_even_with_other_paid_settlements(database, migrated, head):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn)
        assert _reserve(conn, lease, SECOND_CALL)['status'] == 'reserved'
        if head in ('dispatch', 'uncertain'):
            assert _dispatch(conn, lease, SECOND_CALL)['status'] == 'dispatched'
        if head == 'uncertain':
            assert _rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
                        (lease['run_id'], TOKEN_HASH, SECOND_CALL, '{}'))['status'] == 'uncertain'
        elif head in ('recovery', 'refusal'):
            # Conservative negative controls for terminal states that this
            # narrow migration intentionally does not certify as settled.
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute('UPDATE public.lab_arena_ledger SET entry_kind=%s WHERE call_identity=%s',
                            (head, SECOND_CALL))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
        before = _audit(conn, lease)
        assert _expire(conn)['expired'] == 0
        assert _audit(conn, lease) == before


@pytest.mark.parametrize('change', [
    'active', 'execute', 'wrong_runner', 'wrong_round', 'wrong_submission',
    'wrong_miner', 'wrong_assignment', 'wrong_stage', 'wrong_position',
    'wrong_attempt', 'wrong_kind', 'wrong_class', 'wrong_status', 'setup',
    'cleanup', 'secondary_cleanup', 'finished', 'result', 'output',
    'old_stage_generation', 'wrong_lease_generation', 'wrong_claim_generation',
    'no_started', 'wrong_started_generation', 'wrong_started_runner',
    'before_claim', 'after_claim_expiry', 'non_signaled', 'positive_exit',
    'string_exit', 'timed_out', 'missing_timeout', 'null_identity',
    'operator_pause', 'restart_guard',
])
def test_only_exact_terminated_current_score_recovers(database, migrated, change):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, abandoned=change != 'active',
                      kind='execute' if change == 'execute' else 'score')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            columns = {'wrong_runner': ('runner_hotkey', recovery.prior.RUNNER_B),
                       'wrong_round': ('round_id', 'other'), 'wrong_submission': ('submission_id', 'other'),
                       'wrong_miner': ('miner_hotkey', recovery.prior.RUNNER_B),
                       'wrong_assignment': ('assignment_id', 'other'), 'wrong_stage': ('stage', 2),
                       'wrong_position': ('icp_position', 9), 'wrong_attempt': ('attempt', 2),
                       'wrong_kind': ('run_kind', 'execute')}
            content = {'wrong_class': ('error_class', 'OSError'), 'wrong_status': ('status', 'accepted'),
                       'setup': ('failure_stage', 'setup'), 'cleanup': ('failure_stage', 'cleanup'),
                       'non_signaled': ('runtime_host_reason', 'sandbox_launch_failed'),
                       'positive_exit': ('launch_exit_code', 130), 'string_exit': ('launch_exit_code', '-2'),
                       'timed_out': ('launch_timed_out', True)}
            if change in columns:
                column, value = columns[change]
                cur.execute(f'UPDATE public.lab_arena_trajectory_events SET {column}=%s '
                            "WHERE event_kind='runtime.error'", (value,))
            elif change in content:
                key, value = content[change]
                cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,%s::text[],%s::jsonb) WHERE event_kind='runtime.error'",
                            ([key], json.dumps(value)))
            elif change == 'missing_timeout':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET content=content-'launch_timed_out' WHERE event_kind='runtime.error'")
            elif change in ('result', 'output', 'old_stage_generation', 'wrong_lease_generation'):
                clause = {'result': "result_doc='{}'", 'output': "output_ref='present'",
                          'old_stage_generation': 'stage_generation=0',
                          'wrong_lease_generation': 'lease_generation=lease_generation+1'}[change]
                cur.execute(f'UPDATE public.lab_arena_runs SET {clause} WHERE run_id=%s', (lease['run_id'],))
            elif change == 'wrong_claim_generation':
                cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_generation}','99') WHERE run_id=%s", (lease['run_id'],))
            elif change == 'no_started':
                cur.execute("DELETE FROM public.lab_arena_trajectory_events WHERE event_kind='runtime.started'")
            elif change == 'wrong_started_generation':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,'{lease_generation}','99') WHERE event_kind='runtime.started'")
            elif change == 'wrong_started_runner':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET runner_hotkey=%s WHERE event_kind='runtime.started'", (recovery.prior.RUNNER_B,))
            elif change in ('before_claim', 'after_claim_expiry'):
                clause = "occurred_at=now()-interval '1 day'" if change == 'before_claim' else "created_at=now()+interval '1 day'"
                cur.execute(f'UPDATE public.lab_arena_trajectory_events SET {clause} ' "WHERE event_kind='runtime.error'")
            elif change in ('finished', 'secondary_cleanup'):
                cur.execute("UPDATE public.lab_arena_trajectory_events SET event_kind=%s WHERE event_kind='runtime.error'",
                            ('runtime.finished' if change == 'finished' else 'runtime.cleanup_error',))
            elif change == 'null_identity':
                cur.execute('UPDATE public.lab_arena_ledger SET call_identity=NULL')
            elif change == 'operator_pause':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='test'")
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        if change == 'restart_guard':
            generation = restart.guard._acquire(conn, restart._generation(conn))['guard_generation']
            conn.commit()
        before = _audit(conn, lease)
        assert _expire(conn) == {'status': 'ok', 'expired': 0, 'retried': 0}
        assert _audit(conn, lease) == before
        if change == 'restart_guard':
            restart._abort(conn, generation)
        else:
            with conn.cursor() as cur:
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason=''")
            conn.commit()


@pytest.mark.parametrize('kind', ['score', 'execute'])
def test_zero_call_412_recovery_is_unchanged(database, migrated, kind):
    recovery.test_recovery_hands_off_without_ttl_wait_and_preserves_audit(database, migrated, kind)


@pytest.mark.parametrize('winner', ['reservation', 'recovery'])
def test_queued_provider_work_cannot_cross_recovery_fence(database, migrated, winner):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn)
    owner = psycopg.connect(**dsn)
    try:
        if winner == 'reservation':
            assert _reserve(owner, lease, SECOND_CALL, commit=False)['status'] == 'reserved'
        else:
            assert _expire(owner, commit=False)['expired'] == 1
        started = threading.Event()

        def losing_call():
            with psycopg.connect(**dsn) as loser:
                started.set()
                return _expire(loser) if winner == 'reservation' else _reserve(loser, lease, SECOND_CALL)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(losing_call)
            assert started.wait(2)
            with psycopg.connect(**dsn) as observer:
                for _ in range(100):
                    with observer.cursor() as cur:
                        cur.execute("SELECT count(*) FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE 'SELECT public.lab_arena_%'")
                        if cur.fetchone()[0]:
                            break
                    threading.Event().wait(.01)
                else:
                    pytest.fail('reservation/recovery race did not reach the DB lock')
            owner.commit()
            answer = future.result(timeout=5)
        assert answer['expired'] == 0 if winner == 'reservation' else answer['status'] == 'stale'
        # A delayed dispatch uses the same lease fence and cannot spend after recovery.
        if winner == 'recovery':
            assert _dispatch(owner, lease)['status'] == 'stale'
        with owner.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone()[0] == (4 if winner == 'reservation' else 3)
    finally:
        owner.close()
