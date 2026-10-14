"""A closed failed provider call can drain without declaring its bill free."""
import base64
import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena import host_interruption_retry474_postgres_test as prior
from tests.lab_arena import owner_stopped_execute_drain461_postgres_test as stopped
from tests.lab_arena.host_interruption_migration_authority_postgres_test import snapshot

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/476-lab-arena-closed-failed-execute-host-drain.sql'
database = prior.database


@pytest.fixture(scope='module')
def upgraded(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('CREATE ROLE arena_closed_failure_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO arena_closed_failure_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO arena_closed_failure_migrator WITH GRANT OPTION')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE arena_closed_failure_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            cur.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
            cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            schema_acl = cur.fetchone()[0]
            signatures = ('public.lab_arena_expire_leases(text)',
                          'public.lab_arena__restart_drain_state_v1()')
            before = snapshot(cur, signatures)
            cur.execute('SET ROLE arena_closed_failure_migrator')
            cur.execute("SELECT rolsuper,has_schema_privilege(current_user,'public','CREATE'),has_schema_privilege('lab_arena_owner','public','CREATE'),pg_has_role(current_user,'lab_arena_owner','MEMBER') FROM pg_roles WHERE rolname=current_user")
            assert cur.fetchone() == (False, True, False, True)
            cur.execute('COMMIT')
            altered = SQL.read_text().replace(
                'b871e809c70fb12dbacf918e12ca5e75f9d7721f2fdbcb02a3261035cc2d45ff',
                '0' * 64, 1)
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(altered)
            cur.execute('ROLLBACK')
            assert snapshot(cur, signatures) == before
            cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
            assert cur.fetchone()[0] == schema_acl
            first = None
            for _ in range(2):
                cur.execute(SQL.read_text())
                cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                assert cur.fetchone()[0] == schema_acl
                cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'),has_schema_privilege('lab_arena_service','public','CREATE')")
                assert cur.fetchone() == (False, False)
                installed = snapshot(cur, signatures)
                assert all(installed[key][1:] == before[key][1:] for key in signatures)
                if first is None:
                    first = installed
                else:
                    assert installed == first
            cur.execute('RESET ROLE')
    return True


def test_exact_migration_replays_under_production_like_authority(database, upgraded):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT has_function_privilege('anon','public.lab_arena_expire_leases(text)','EXECUTE'),has_function_privilege('lab_arena_service','public.lab_arena_expire_leases(text)','EXECUTE')")
            assert cur.fetchone() == (False, True)


def _closed_failure(conn, *, terminal_status=422, mutation=None, old_day=False,
                    open_dispatch=False, signed_host=True):
    lease = stopped._seed(conn, settled=1, abandoned=signed_host)
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET evaluation_date=%s WHERE round_id=%s",
                    ('2026-10-13' if old_day else '2026-10-14',
                     stopped.paid.recovery.prior.ROUND))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    call = 'sha256:' + hashlib.sha256(b'closed-failed-execute').hexdigest()
    request = 'ctx-tool-' + call[7:39]
    execution = 'arena:' + call[7:]
    credential = 'sha256:' + 'c' * 64
    reservation = {
        'request_hash': 'sha256:' + 'd' * 64,
        'deepline_request_id': request,
        'deepline_execution_key': execution,
        'credential_fingerprint': credential,
        'tool': 'exa_contents',
        'deepline_billing_provider': 'exa',
    }
    assert stopped.paid._rpc(conn,
        "lab_arena_reserve_call(%s,%s,%s,'exa.contents','deepline','host',0,%s::jsonb,4500)",
        (lease['run_id'], stopped.paid.TOKEN_HASH, call, json.dumps(reservation)))['status'] == 'reserved'
    assert stopped.paid._dispatch(conn, lease, call)['status'] == 'dispatched'
    if open_dispatch:
        return lease, call
    terminal = {'status': terminal_status, 'call_succeeded': False, 'headers': {},
                'body_b64': base64.b64encode(b'{"error":{"code":"provider_request_rejected"}}').decode()}
    doc = {'reason': 'missing_provider_cost', 'call_succeeded': False,
           'provider_status': terminal_status, 'deepline_request_id': request,
           'deepline_execution_key': execution, 'credential_fingerprint': credential,
           'deepline_operation': 'exa_contents', 'deepline_terminal_response': terminal}
    if mutation == 'missing_terminal':
        doc.pop('deepline_terminal_response')
    elif mutation == 'wrong_request':
        doc['deepline_request_id'] = 'ctx-tool-' + 'f' * 32
    elif mutation == 'wrong_execution':
        doc['deepline_execution_key'] = 'arena:' + 'f' * 64
    elif mutation == 'wrong_credential':
        doc['credential_fingerprint'] = 'sha256:' + 'f' * 64
    elif mutation == 'wrong_operation':
        doc['deepline_operation'] = 'other_tool'
    elif mutation == 'async_job':
        doc['deepline_async_job_ids'] = ['job-1']
    elif mutation == 'success':
        doc['call_succeeded'] = True
    elif mutation == 'terminal_success':
        terminal['call_succeeded'] = True
    elif mutation == 'mismatched_status':
        terminal['status'] = 404
    assert stopped.paid._rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
        (lease['run_id'], stopped.paid.TOKEN_HASH, call, json.dumps(doc)))['status'] == 'uncertain'
    return lease, call


@pytest.mark.parametrize('terminal_status', [400, 403, 404, 422])
def test_bound_terminal_failure_drains_without_pricing_the_bill(database, upgraded,
                                                                 terminal_status):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        stopped._reset(conn)
        lease, _ = _closed_failure(conn, terminal_status=terminal_status)
        before = stopped.paid._audit(conn, lease)
        generation = stopped.prior._hold_capture(conn)
        drained = stopped.guard._quiescence(conn, generation=generation)
        assert drained['expired_receipt_count'] == 1, drained
        assert drained['pending_retry_count'] == 1, drained
        after = stopped.paid._audit(conn, lease)
        assert after['lab_arena_ledger'] == before['lab_arena_ledger']
        assert after['run'][2]['recovery_reason'] == 'authenticated_closed_failed_execute_runtime_host_error'
        assert stopped.guard._quiescence(conn, generation=generation)['outcome_commitment'] == drained['outcome_commitment']
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s',
                        (lease['assignment_id'],))
            assert cur.fetchone()[0] == 2


def test_closed_failure_retry_accepts_and_late_bill_settles_once(database, upgraded):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        stopped._reset(conn)
        lease, call = _closed_failure(conn)
        generation = stopped.prior._hold_capture(conn)
        drained = stopped.guard._quiescence(conn, generation=generation)
        assert drained['preserved'] and drained['pending_retry_count'] == 1
        stopped.guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1',
                           stopped.guard.GUARD, stopped.guard.OWNER, generation,
                           'gateway_destructive')
        stopped.guard._rpc(conn, 'lab_arena_mark_restart_ready_v1',
                           stopped.guard.GUARD, stopped.guard.OWNER, generation,
                           'gateway_ready')
        released = stopped.guard._rpc(conn, 'lab_arena_release_restart_guard_v1',
                                      stopped.guard.GUARD, stopped.guard.OWNER,
                                      generation, 'closed-failure-test')
        assert released['guard_present'] is False
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,"
                        "pause_reason='' WHERE singleton")
        conn.commit()
        retry = stopped.paid.recovery._claim(conn, stopped.paid.recovery.prior.RUNNER_B, 'b')
        assert retry['status'] == 'leased', retry
        with conn.cursor() as cur:
            cur.execute('SELECT attempt FROM public.lab_arena_runs WHERE run_id=%s', (retry['run_id'],))
            assert cur.fetchone()[0] == 2
        second_call = 'sha256:' + hashlib.sha256(b'closed-failure-retry').hexdigest()
        retry_token = 'sha256:' + 'b' * 64
        assert stopped.paid._reserve(conn, retry, second_call, token_hash=retry_token)['status'] == 'reserved'
        assert stopped.paid._dispatch(conn, retry, second_call, token_hash=retry_token)['status'] == 'dispatched'
        assert stopped.paid._settle(conn, retry, second_call, token_hash=retry_token)['status'] == 'settled'
        assert prior.finish(conn, retry, 'accepted', token='b')['status'] == 'accepted'
        with conn.cursor() as cur:
            cur.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger "
                        "WHERE run_id IN (%s,%s) AND entry_kind='settlement'",
                        (lease['run_id'], retry['run_id']))
            assert cur.fetchone() == (2, 200)
            cur.execute("SELECT entry_id,entry_doc #>> '{call,deepline_request_id}',"
                        "entry_doc #>> '{call,deepline_execution_key}',"
                        "entry_doc #>> '{call,credential_fingerprint}' "
                        "FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'", (call,))
            entry_id, request, execution, credential = cur.fetchone()
        args = (stopped.paid.recovery.prior.ROUND, lease['run_id'], call,
                entry_id, request, 'exa_contents', credential, 100, '0.001',
                execution, 'exa::contents-closed-failure-test')
        first = stopped.paid._rpc(conn,
            'lab_arena_reconcile_deepline_cost_v2(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', args)
        assert first['status'] == 'settled', first
        repeat = stopped.paid._rpc(conn,
            'lab_arena_reconcile_deepline_cost_v2(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', args)
        assert repeat['idempotent'] is True, repeat
        with conn.cursor() as cur:
            cur.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger "
                        "WHERE call_identity=%s AND entry_kind='settlement'", (call,))
            assert cur.fetchone() == (1, 100)
            cur.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger "
                        "WHERE run_id IN (%s,%s) AND entry_kind='settlement'",
                        (lease['run_id'], retry['run_id']))
            assert cur.fetchone() == (3, 300)
            cur.execute('SELECT public.lab_arena__execute_assignment_calls_v1(%s,%s)',
                        (lease['assignment_id'], 'deepline'))
            assert cur.fetchone()[0] == 3


@pytest.mark.parametrize('variant', [
    'missing_terminal', 'wrong_request', 'wrong_execution', 'wrong_credential',
    'wrong_operation', 'async_job', 'success', 'terminal_success',
    'mismatched_status', 'status_0', 'status_200', 'status_202', 'status_500',
    'status_502', 'open_dispatch', 'old_day', 'no_host',
])
def test_unproved_or_open_call_remains_held(database, upgraded, variant):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        stopped._reset(conn)
        lease, _ = _closed_failure(conn,
            terminal_status={'status_0': 0, 'status_200': 200, 'status_202': 202,
                             'status_500': 500, 'status_502': 502}.get(variant, 422),
            mutation=variant, old_day=variant == 'old_day',
            open_dispatch=variant == 'open_dispatch',
            signed_host=variant != 'no_host')
        before = stopped.paid._audit(conn, lease)
        generation = stopped.prior._hold_capture(conn)
        drained = stopped.guard._quiescence(conn, generation=generation)
        assert (drained['still_leased_count'], drained['expired_receipt_count'],
                drained['pending_retry_count']) == (1, 0, 0), (variant, drained)
        assert stopped.paid._audit(conn, lease) == before


def test_existing_acceptance_publication_and_rewards_still_work(database, upgraded, tmp_path):
    with database[0].connect(**database[1]) as conn:
        stopped._reset(conn)
    prior.test_current_schema_accepted_scores_publish_and_activate_reward(database, tmp_path)


def test_old_baseline_publication_is_unchanged(database, upgraded, tmp_path):
    with database[0].connect(**database[1]) as conn:
        stopped._reset(conn)
    prior.test_current_schema_model_zero_preserves_baseline_scores_and_publication(database, tmp_path)
