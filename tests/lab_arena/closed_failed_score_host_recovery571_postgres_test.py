"""Closed failed score calls release stopped workers without pricing their bills."""
import base64
import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena import expiry_recovery_candidate552_postgres_test as previous
from tests.lab_arena import closed_failed_execute_drain476_postgres_test as failed
from tests.lab_arena import quiescence_recovery_gate522_postgres_test as gate

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/571-lab-arena-closed-failed-score-host-recovery.sql'
SIGNATURES = ('public.lab_arena_expire_leases(text)',
              'public.lab_arena__restart_drain_state_v1()')
PRE = ('5025e77ac2a602941e7166d0a201ba49efb03d0da8af1a0fda1e1889056de40d',
       'a38782b4b6555c455fae64a2434d0bbd394f2f970bc42326475f187a4a1756b8')
POST = ('cbe667688a31c3e0fbc39e968721b92ff206780c6818df8afde4eeaedd98c187',
        '6de3a3673357bbe823992075fd00443247e1b19c50e7cc3d01307ea381eab984')
paid = failed.stopped.paid
cohort = previous.prior.prior


def definitions(cur):
    result = []
    for signature in SIGNATURES:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        result.append(cur.fetchone()[0])
    return result


@pytest.fixture(scope='module')
def database():
    gen = previous.database.__wrapped__()
    db = next(gen)
    try:
        # Replay canonical migrations, including522/523/527/551/552. Check
        # exact parity with both affected production functions before testing.
        with db[0].connect(**db[1]) as conn:
            previous.apply(conn)
            with conn.cursor() as cur:
                assert tuple(hashlib.sha256(s.encode()).hexdigest()
                             for s in definitions(cur)) == PRE
        yield db
    finally:
        gen.close()


@pytest.fixture(autouse=True)
def restore(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        originals = definitions(cur)
    yield
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for source in originals:
            cur.execute(source)
        cur.execute('TRUNCATE public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        cur.execute('TRUNCATE public.lab_arena_rounds CASCADE')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,"
                    "pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',"
                    "guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")


def apply(conn):
    with conn.cursor() as cur:
        cur.execute(SQL.read_text())
    conn.commit()


def seed(database, conn, *, kind='score', status=403, count=1, mutation=None, attempt=1):
    if kind == 'score':
        lease = cohort.cohort(conn, count=1, extra=0, started=1)[0]
    else:
        with gate.legacy_floor_zero(database):
            lease = paid._seed(conn, kind='execute', abandoned=False, settled=False)
        # The historical fixture predates the bounded closed-failure proof.
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date='2026-10-20'")
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
    if kind == 'score' and attempt > 1:
        # Reach late attempts through ordinary completion and claim authority.
        # Only the fixture clock/backoff is shortened; paid RPCs retain all gates.
        with conn.cursor() as cur:
            cur.execute('SELECT round_id,runner_hotkey FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            round_id, runner = cur.fetchone()
        for number in range(2, attempt + 1):
            result = paid._rpc(conn, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
                (lease['run_id'], paid.TOKEN_HASH, json.dumps({'terminal_status': 'judge_error'}),
                 'judge_error', None))
            assert result['confirmation_attempt'] == number
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'", (round_id,))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            lease = paid._rpc(conn, 'lab_arena_claim_assignment(%s,%s,7,20,%s,%s,%s,%s,4500)',
                (round_id, runner, [], format(number, '032x'),
                 'sha256:' + format(number, '064x'), paid.TOKEN_HASH))
            assert (lease['status'], lease['attempt']) == ('leased', number)
        paid._event(conn, lease, 'runtime.started',
            {'status': 'starting', 'runtime': 'runsc', 'lease_generation': lease['lease_generation']})
    with conn.cursor() as cur:
        cur.execute('SELECT round_id,assignment_id FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
        lease = dict(lease, **dict(zip(('round_id', 'assignment_id'), cur.fetchone())))
    funding = paid._rpc(conn, 'lab_arena_provider_funding(%s,%s)',
        (lease['run_id'], 'deepline'))['funding_source']
    if kind == 'score':
        assert paid._rpc(conn,
            "lab_arena_reserve_call(%s,%s,%s,'exa.contents','deepline',%s,0,'{}'::jsonb,4500)",
            (lease['run_id'], paid.TOKEN_HASH, paid.CALL, funding))['status'] == 'reserved'
        assert paid._dispatch(conn, lease)['status'] == 'dispatched'
    calls = []
    for index in range(count):
        call = 'sha256:' + hashlib.sha256(f'closed-failed-score-{index}'.encode()).hexdigest()
        request, execution, credential = 'ctx-tool-' + call[7:39], 'arena:' + call[7:], 'sha256:' + 'c' * 64
        reservation = {'request_hash': 'sha256:' + 'd' * 64,
            'deepline_request_id': request, 'deepline_execution_key': execution,
            'credential_fingerprint': credential, 'tool': 'scrapingdog_scrape',
            'deepline_billing_provider': 'scrapingdog'}
        assert paid._rpc(conn,
            "lab_arena_reserve_call(%s,%s,%s,'scrapingdog.scrape','deepline',%s,0,%s::jsonb,4500)",
            (lease['run_id'], paid.TOKEN_HASH, call, funding, json.dumps(reservation)))['status'] == 'reserved'
        if mutation == 'reservation':
            calls.append(call)
            continue
        assert paid._dispatch(conn, lease, call)['status'] == 'dispatched'
        if mutation == 'dispatch':
            calls.append(call)
            continue
        terminal = {'status': status, 'call_succeeded': False, 'headers': {},
            'body_b64': base64.b64encode(b'{"error":{"code":"provider_request_rejected"}}').decode()}
        doc = {'reason': 'missing_provider_cost', 'call_succeeded': False,
            'provider_status': status, 'top_level_job_status': 'failed',
            'deepline_request_id': request, 'deepline_execution_key': execution,
            'credential_fingerprint': credential, 'deepline_operation': 'scrapingdog_scrape',
            'deepline_terminal_response': terminal}
        changes = {'request': ('deepline_request_id', 'ctx-tool-' + 'f' * 32),
            'execution': ('deepline_execution_key', 'arena:' + 'f' * 64),
            'credential': ('credential_fingerprint', 'sha256:' + 'f' * 64),
            'operation': ('deepline_operation', 'other'),
            'async_job': ('deepline_async_job_ids', ['still-running']),
            'success': ('call_succeeded', True), 'untyped_success': ('call_succeeded', 'false'),
            'untyped_status': ('provider_status', str(status))}
        if mutation in changes:
            key, value = changes[mutation]
            doc[key] = value
        elif mutation == 'missing_terminal':
            doc.pop('deepline_terminal_response')
        elif mutation == 'terminal_success':
            terminal['call_succeeded'] = True
        elif mutation == 'mismatched_status':
            terminal['status'] = 404
        elif mutation == 'empty_body':
            terminal['body_b64'] = ''
        assert paid._rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
            (lease['run_id'], paid.TOKEN_HASH, call, json.dumps(doc)))['status'] == 'uncertain'
        calls.append(call)
    if mutation != 'no_error':
        paid._event(conn, lease, 'runtime.error', dict(gate.HOST_ERROR))
    # A gateway settlement after abandonment renews the live deadline. The
    # recovery must retain the frozen claim and authenticate its original window.
    assert paid._settle(conn, lease)['status'] == 'settled'
    if mutation in ('no_start', 'cleanup', 'wrong_generation'):
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if mutation == 'no_start':
                cur.execute("DELETE FROM public.lab_arena_trajectory_events WHERE event_kind='runtime.started'")
            elif mutation == 'cleanup':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET event_kind='runtime.cleanup_error' WHERE event_kind='runtime.error'")
            else:
                cur.execute('UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s', (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
    return lease, calls


def expire(conn, lease):
    return paid._rpc(conn, 'lab_arena_expire_leases(%s)', (lease['round_id'],))


def preserved(cur):
    # Complete frozen rounds, accepted outputs and caches are outside the fix.
    result = []
    for table, where in [('lab_arena_rounds', ''), ('lab_arena_judgment_cache', ''),
                         ('lab_arena_runs', " WHERE status='accepted'")]:
        cur.execute(f"SELECT md5(COALESCE(jsonb_agg(to_jsonb(r) ORDER BY to_jsonb(r)::text)::text,'null')) FROM public.{table} r{where}")
        result.append(cur.fetchone()[0])
    return result


@pytest.mark.parametrize('status', [400, 403, 404, 422])
@pytest.mark.parametrize('owner', [False, True])
def test_score_failure_reproduces_before_and_recovers_once_after(database, status, owner):
    with database[0].connect(**database[1]) as conn:
        lease, _ = seed(database, conn, status=status, count=5 if status == 403 else 1)
        before = paid._audit(conn, lease)
        with conn.cursor() as cur:
            frozen = preserved(cur)
        generation = failed.stopped.prior._hold_capture(conn) if owner else None
        if owner:
            old = failed.stopped.guard._quiescence(conn, generation=generation)
            assert (old['still_leased_count'], old['expired_receipt_count']) == (1, 0)
        else:
            assert expire(conn, lease) == {'status': 'ok', 'expired': 0, 'retried': 0}
        apply(conn)
        if owner:
            assert expire(conn, lease)['expired'] == 0  # The guard still holds unowned callers.
            result = failed.stopped.guard._quiescence(conn, generation=generation)
            assert (result['expired_receipt_count'], result['pending_retry_count'], result['preserved']) == (1, 1, True)
            repeat = failed.stopped.guard._quiescence(conn, generation=generation)
            assert repeat['outcome_commitment'] == result['outcome_commitment']
        else:
            assert expire(conn, lease) == {'status': 'ok', 'expired': 1, 'retried': 1}
            assert expire(conn, lease) == {'status': 'ok', 'expired': 0, 'retried': 0}
        after = paid._audit(conn, lease)
        assert after['lab_arena_ledger'] == before['lab_arena_ledger']
        assert after['lab_arena_trajectory_events'] == before['lab_arena_trajectory_events']
        assert after['run'][:2] == ('failed', 'lease_expired')
        assert after['run'][2]['recovery_reason'] == 'authenticated_closed_failed_score_runtime_host_error'
        assert after['run'][3] == before['run'][3]  # Frozen claim remains exact.
        with conn.cursor() as cur:
            assert preserved(cur) == frozen
            cur.execute('SELECT attempt,status,scored_run_id,company_judgment_refs FROM public.lab_arena_runs WHERE assignment_id=%s ORDER BY attempt', (lease['assignment_id'],))
            rows = cur.fetchall()
            assert len(rows) == 2 and rows[-1][:2] == (2, 'pending')
            assert rows[0][2:] == rows[1][2:]


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('status', [400, 403, 404, 422])
def test_existing_execute_closed_failure_is_unchanged(database, installed, status):
    with database[0].connect(**database[1]) as conn:
        lease, _ = seed(database, conn, kind='execute', status=status)
        before = paid._audit(conn, lease)
        if installed:
            apply(conn)
        assert expire(conn, lease) == {'status': 'ok', 'expired': 1, 'retried': 1}
        assert expire(conn, lease) == {'status': 'ok', 'expired': 0, 'retried': 0}
        after = paid._audit(conn, lease)
        assert after['run'][2]['recovery_reason'] == 'authenticated_closed_failed_execute_runtime_host_error'
        assert after['lab_arena_ledger'] == before['lab_arena_ledger']


@pytest.mark.parametrize('owner', [False, True])
@pytest.mark.parametrize('mutation,status', [
    ('no_error', 403), ('no_start', 403), ('cleanup', 403), ('wrong_generation', 403),
    ('reservation', 403), ('dispatch', 403), ('async_job', 403),
    ('request', 403), ('execution', 403), ('credential', 403), ('operation', 403),
    ('missing_terminal', 403), ('terminal_success', 403), ('mismatched_status', 403),
    ('empty_body', 403), ('success', 403), ('untyped_success', 403), ('untyped_status', 403),
    (None, 0), (None, 200), (None, 202), (None, 500), (None, 502),
])
def test_live_unbound_or_ambiguous_call_remains_held(database, owner, mutation, status):
    with database[0].connect(**database[1]) as conn:
        lease, _ = seed(database, conn, status=status, mutation=mutation)
        apply(conn)
        generation = failed.stopped.prior._hold_capture(conn) if owner else None
        before = paid._audit(conn, lease)
        if owner:
            result = failed.stopped.guard._quiescence(conn, generation=generation)
            assert (result['still_leased_count'], result['expired_receipt_count'], result['pending_retry_count']) == (1, 0, 0)
        else:
            assert expire(conn, lease) == {'status': 'ok', 'expired': 0, 'retried': 0}
        assert paid._audit(conn, lease) == before


def test_late_actual_bill_reconciles_once_after_score_recovery(database):
    with database[0].connect(**database[1]) as conn:
        lease, calls = seed(database, conn)
        apply(conn)
        assert expire(conn, lease)['retried'] == 1
        with conn.cursor() as cur:
            cur.execute("SELECT entry_id,entry_doc #>> '{call,deepline_request_id}',entry_doc #>> '{call,deepline_execution_key}',entry_doc #>> '{call,credential_fingerprint}' FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'", (calls[0],))
            entry, request, execution, credential = cur.fetchone()
        args = (lease['round_id'], lease['run_id'], calls[0], entry, request,
                'scrapingdog_scrape', credential, 2000, '0.02', execution,
                'scrapingdog::scrape-closed-score-test')
        expression = 'lab_arena_reconcile_deepline_cost_v2(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        assert paid._rpc(conn, expression, args)['status'] == 'settled'
        assert paid._rpc(conn, expression, args)['idempotent'] is True
        with conn.cursor() as cur:
            cur.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'", (calls[0],))
            assert cur.fetchone() == (1, 2000)


def test_migration_exact_replay_changes_only_two_functions_without_rows(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        seed(database, conn)
        catalog = previous.prior.catalog(cur)
        cur.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r')
        rows = cur.fetchone()[0]
        cur.execute('CREATE ROLE closed_score571_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO closed_score571_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO closed_score571_migrator')
        cur.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
        cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
        assert cur.fetchone() == (False,)
        cur.execute('SET ROLE closed_score571_migrator')
        cur.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert cur.fetchone() == (False,)
        apply(conn)
        apply(conn)
        cur.execute('RESET ROLE')
        after = previous.prior.catalog(cur)
        changed = [oid for oid in catalog if catalog[oid] != after[oid]]
        assert len(changed) == 2
        for oid in changed:
            assert after[oid][1] == catalog[oid][1]
        assert tuple(hashlib.sha256(s.encode()).hexdigest() for s in definitions(cur)) == POST
        cur.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r')
        assert cur.fetchone()[0] == rows


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('target', SIGNATURES)
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'path', 'security', 'volatility'])
def test_migration_rejects_source_or_security_drift(database, installed, target, drift):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if installed:
            apply(conn)
        cur.execute('SAVEPOINT hostile')
        if drift == 'source':
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (target,))
            cur.execute(cur.fetchone()[0].replace('BEGIN\n', 'BEGIN\n  NULL;\n', 1))
        else:
            cur.execute({'owner': 'ALTER FUNCTION ' + target + ' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION ' + target + ' TO anon',
                'path': 'ALTER FUNCTION ' + target + ' SET search_path=public,pg_catalog',
                'security': 'ALTER FUNCTION ' + target + ' SECURITY INVOKER',
                'volatility': 'ALTER FUNCTION ' + target + (' STABLE' if target == SIGNATURES[0] else ' VOLATILE')}[drift])
        body = SQL.read_text().split('DO $closed_failed_score_recovery571$')[1].split('$closed_failed_score_recovery571$;')[0]
        with pytest.raises(database[0].Error):
            cur.execute('DO $test$' + body + '$test$')
        cur.execute('ROLLBACK TO SAVEPOINT hostile')


def test_five_failed_calls_on_attempt16_create_only_attempt17(database):
    with database[0].connect(**database[1]) as conn:
        lease, calls = seed(database, conn, count=5, attempt=16)
        before = paid._audit(conn, lease)
        with conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE assignment_id=%s AND attempt<16 ORDER BY attempt", (lease['assignment_id'],))
            history = cur.fetchall()
        assert len(history) == 15 and len(calls) == 5
        assert expire(conn, lease) == {'status': 'ok', 'expired': 0, 'retried': 0}
        apply(conn)
        assert expire(conn, lease) == {'status': 'ok', 'expired': 1, 'retried': 1}
        assert expire(conn, lease) == {'status': 'ok', 'expired': 0, 'retried': 0}
        after = paid._audit(conn, lease)
        assert after['lab_arena_ledger'] == before['lab_arena_ledger']
        assert after['lab_arena_trajectory_events'] == before['lab_arena_trajectory_events']
        with conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE assignment_id=%s AND attempt<16 ORDER BY attempt", (lease['assignment_id'],))
            assert cur.fetchall() == history
            cur.execute("SELECT attempt,status,scored_run_id,company_judgment_refs FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>=16 ORDER BY attempt", (lease['assignment_id'],))
            rows = cur.fetchall()
            assert [r[:2] for r in rows] == [(16, 'failed'), (17, 'pending')]
            assert rows[0][2:] == rows[1][2:]
