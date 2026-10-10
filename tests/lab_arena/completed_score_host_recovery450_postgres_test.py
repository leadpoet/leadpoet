"""Completed research can release a stopped scorer without settling its bill."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading

import pytest

from tests.lab_arena import settled_execute_host_recovery442_postgres_test as prior

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/450-lab-arena-completed-score-host-recovery.sql'
PREHASH = '9aa15b0f1aa66e61a668c708384600954b28bef7ef530781b0a25da357d40c71'
database = prior.database
CREDENTIAL = 'sha256:' + 'c' * 64
REQUEST_HASH = 'sha256:' + 'd' * 64
NATIVE = 'public-execute:completed-fixture'


@pytest.fixture(scope='module')
def migrated(database):
    prior.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            security = prior.identity._security(cur)
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (prior.identity.EXPIRY,))
            before = cur.fetchone()[0]
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREHASH, '0' * 64, 1))
            cur.execute('ROLLBACK')
            assert prior.identity._hash(cur, prior.identity.EXPIRY) == PREHASH
            cur.execute(SQL.read_text())
            post = prior.identity._hash(cur, prior.identity.EXPIRY)
            cur.execute(SQL.read_text())
            assert prior.identity._hash(cur, prior.identity.EXPIRY) == post
            assert prior.identity._security(cur) == security
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (prior.identity.EXPIRY,))
            after = cur.fetchone()[0]
            # Only the early paid-head predicate and its truthful reason change.
            prefix = '       AND NOT EXISTS (\n         SELECT 1 FROM (\n           SELECT DISTINCT ON (cost.call_identity)'
            tail = '  FOR v_run IN\n    SELECT * FROM public.lab_arena_runs\n    WHERE round_id = p_round_id AND status = \'leased\''
            assert before.split(prefix)[0] == after.split(prefix)[0]
            assert before.split(tail)[1] == after.split(tail)[1]
    return True


def _unpriced(conn, lease, call=prior.SECOND_CALL):
    native = NATIVE + '-' + call[-8:]
    request = 'ctx-tool-' + call[7:39]
    execution = 'arena:' + call[7:]
    doc = {'request_hash': REQUEST_HASH, 'deepline_request_id': request,
           'deepline_execution_key': execution, 'credential_fingerprint': CREDENTIAL,
           'tool': 'exa_contents', 'deepline_billing_provider': 'exa'}
    assert prior._rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'exa.contents','deepline','host',0,%s::jsonb,4500)",
                      (lease['run_id'], prior.TOKEN_HASH, call, json.dumps(doc)))['status'] == 'reserved'
    assert prior._dispatch(conn, lease, call)['status'] == 'dispatched'
    content = {'reason': 'missing_provider_cost', 'call_succeeded': True,
               'provider_status': 200, 'top_level_job_status': 'completed',
               'deepline_request_id': request, 'deepline_execution_key': execution,
               'credential_fingerprint': CREDENTIAL, 'deepline_operation': 'exa_contents',
               'deepline_job_id': native}
    assert prior._rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
                      (lease['run_id'], prior.TOKEN_HASH, call, json.dumps(content)))['status'] == 'uncertain'
    terminal = {'status': 200, 'call_succeeded': True, 'headers': {},
                'body_b64': base64.b64encode(b'{"results":[{"url":"https://example.com"}]}').decode()}
    assert prior._rpc(conn, 'lab_arena_recover_deepline_response_v1(%s,%s,%s,%s,%s,%s,%s,%s,NULL,%s::jsonb,4500)',
                      (lease['run_id'], prior.TOKEN_HASH, call, REQUEST_HASH, execution,
                       CREDENTIAL, native, 'exa_contents', json.dumps(terminal)))['status'] == 'uncertain'
    return dict(call=call, native=native, execution=execution)


def _seed(conn, *, count=1, kind='score'):
    lease = prior._seed(conn, abandoned=False, kind=kind)
    with conn.cursor() as cur:
        # Only the disposable fixture changes its frozen policy. Production
        # receives the function-only migration and retains its source costs.
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                    "jsonb_set(configuration_doc || %s::jsonb,"
                    "'{scorer_policy,scoring_adapter_version}',"
                    "'\"qualification_integrity_v2\"'::jsonb),"
                    "confirmation_bank_ref='arena/fixture/confirmation.json',"
                    "confirmation_bank_hash=%s", (json.dumps({
                        'sourcing_cost_eligibility_policy': 'successful_calls_per_icp_v1',
                        'integrity_policy': 'arena_integrity_v1',
                        'execution_icp_cap_microusd': 4000000,
                        'cost_per_company_microusd': 800000}), 'sha256:' + 'a' * 64))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    calls = []
    for index in range(count):
        call = 'sha256:' + hashlib.sha256(('completed-' + str(index)).encode()).hexdigest()
        calls.append(_unpriced(conn, lease, call))
    prior._event(conn, lease, 'runtime.error', {
        'status': 'abandoned', 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError',
        'runtime_host_reason': 'sandbox_launcher_signaled', 'launch_exit_code': -2,
        'launch_timed_out': False})
    return lease, calls


def _cache(conn, lease):
    with conn.cursor() as cur:
        cur.execute('SELECT coalesce(jsonb_agg(to_jsonb(x) ORDER BY call_identity),\'[]\') '
                    'FROM public.lab_arena_deepline_call_responses x WHERE run_id=%s', (lease['run_id'],))
        return cur.fetchone()[0]


@pytest.mark.parametrize('count', [3, 5, 4, 0, 4, 1, 7, 0, 3, 3, 3, 1])
def test_twelve_evidence_shaped_leases_preserve_bills_and_cached_results(database, migrated, count):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease, calls = _seed(conn, count=count)
        before, cache = prior._audit(conn, lease), _cache(conn, lease)
        args = (prior.recovery.prior.ROUND, lease['submission_id'], lease['icp_position'], 1)
        cost = prior._rpc(conn, 'lab_arena_icp_cost_eligibility(%s,%s,%s,%s)', args)
        assert prior._expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 1}
        assert prior._expire(conn) == {'status': 'ok', 'expired': 0, 'retried': 0}
        after = prior._audit(conn, lease)
        assert after['lab_arena_ledger'] == before['lab_arena_ledger']
        assert after['lab_arena_trajectory_events'] == before['lab_arena_trajectory_events']
        assert _cache(conn, lease) == cache
        assert prior._rpc(conn, 'lab_arena_icp_cost_eligibility(%s,%s,%s,%s)', args) == cost
        assert after['run'][0:2] == ('failed', 'lease_expired')
        reason = ('authenticated_completed_unpriced_score_runtime_host_error' if count
                  else 'authenticated_settled_score_runtime_host_error')
        assert after['run'][2]['recovery_reason'] == reason
        assert after['run'][3] == lease
        if calls:
            # Existing late accounting still accepts the original uncertain
            # identity after early expiry, once and without changing any price.
            call = calls[0]
            with conn.cursor() as cur:
                cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'", (call['call'],))
                entry = cur.fetchone()[0]
            arguments = (prior.recovery.prior.ROUND, lease['run_id'], call['call'], entry,
                         call['native'], 'exa_contents', CREDENTIAL, 2000, '0.02',
                         call['execution'], call['native'])
            expression = 'lab_arena_reconcile_deepline_cost_v2(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
            assert prior._rpc(conn, expression, arguments)['status'] == 'settled'
            assert prior._rpc(conn, expression, arguments)['status'] == 'settled'
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'", (call['call'],))
                assert cur.fetchone()[0] == 1


@pytest.mark.parametrize('change', [
    'execute', 'missing_cache', 'cache_run', 'cache_reservation', 'cache_request',
    'cache_status', 'cache_status_string', 'cache_success', 'cache_success_string',
    'cache_headers', 'cache_body', 'cache_empty_body', 'credential', 'request',
    'execution', 'operation', 'native', 'running_job', 'missing_job_status',
    'failed_call', 'string_success', 'provider_status', 'string_status',
    'uncertain_outcome', 'missing_binding', 'reservation', 'dispatch',
    'missing_dispatch', 'dispatch_run', 'cleanup', 'finished', 'generation',
    'claim_generation', 'stage_inactive', 'operator_pause', 'restart_guard',
])
def test_incomplete_or_unbound_calls_still_block(database, migrated, change):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease, calls = _seed(conn, kind='execute' if change == 'execute' else 'score')
        call = calls[0]['call']
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if change == 'missing_cache':
                cur.execute('DELETE FROM public.lab_arena_deepline_call_responses WHERE call_identity=%s', (call,))
            elif change in ('cache_run', 'cache_reservation', 'cache_request'):
                clause = {'cache_run': "run_id='foreign'", 'cache_reservation': 'reservation_entry_id=reservation_entry_id+1',
                          'cache_request': "request_id='foreign'"}[change]
                cur.execute('UPDATE public.lab_arena_deepline_call_responses SET ' + clause + ' WHERE call_identity=%s', (call,))
            elif change.startswith('cache_'):
                key, value = {'cache_status': ('status', 202), 'cache_status_string': ('status', '200'),
                    'cache_success': ('call_succeeded', False), 'cache_success_string': ('call_succeeded', 'true'),
                    'cache_headers': ('headers', []), 'cache_body': ('body_b64', None),
                    'cache_empty_body': ('body_b64', '')}[change]
                cur.execute('UPDATE public.lab_arena_deepline_call_responses SET terminal_response=jsonb_set(terminal_response,%s::text[],%s::jsonb) WHERE call_identity=%s', ([key], json.dumps(value), call))
            elif change in ('credential', 'request', 'execution', 'operation', 'native', 'running_job',
                            'failed_call', 'string_success', 'provider_status', 'string_status', 'uncertain_outcome'):
                key, value = {'credential': ('credential_fingerprint', 'sha256:' + 'b' * 64),
                    'request': ('deepline_request_id', 'ctx-tool-' + 'b' * 32),
                    'execution': ('deepline_execution_key', 'arena:' + 'b' * 64),
                    'operation': ('deepline_operation', 'other'), 'native': ('deepline_job_id', 'other'),
                    'running_job': ('top_level_job_status', 'running'), 'failed_call': ('call_succeeded', False),
                    'string_success': ('call_succeeded', 'true'), 'provider_status': ('provider_status', 202),
                    'string_status': ('provider_status', '200'), 'uncertain_outcome': ('reason', 'transport_failure')}[change]
                cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,%s::text[],%s::jsonb) WHERE call_identity=%s AND entry_kind='uncertain'", (['call', key], json.dumps(value), call))
            elif change == 'missing_job_status':
                cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc#-'{call,top_level_job_status}' WHERE call_identity=%s AND entry_kind='uncertain'", (call,))
            elif change == 'missing_binding':
                cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc-'credential_fingerprint' WHERE call_identity=%s AND entry_kind='reservation'", (call,))
            elif change in ('reservation', 'dispatch'):
                cur.execute("DELETE FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'", (call,))
                if change == 'reservation':
                    cur.execute("DELETE FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='dispatch'", (call,))
            elif change == 'missing_dispatch':
                cur.execute("DELETE FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='dispatch'", (call,))
            elif change == 'dispatch_run':
                cur.execute("UPDATE public.lab_arena_ledger SET run_id='foreign' WHERE call_identity=%s AND entry_kind='dispatch'", (call,))
            elif change in ('cleanup', 'finished'):
                cur.execute("UPDATE public.lab_arena_trajectory_events SET event_kind=%s WHERE event_kind='runtime.error'", ('runtime.cleanup_error' if change == 'cleanup' else 'runtime.finished',))
            elif change == 'generation':
                cur.execute('UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s', (lease['run_id'],))
            elif change == 'claim_generation':
                cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_generation}','99') WHERE run_id=%s", (lease['run_id'],))
            elif change == 'stage_inactive':
                cur.execute("UPDATE public.lab_arena_rounds SET status='stage1_closed'")
            elif change == 'operator_pause':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='test'")
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        generation = None
        if change == 'restart_guard':
            generation = prior.restart.guard._acquire(conn, prior.restart._generation(conn))['guard_generation']
            conn.commit()
        before, cache = prior._audit(conn, lease), _cache(conn, lease)
        assert prior._expire(conn) == {
            'status': 'no_stage' if change == 'stage_inactive' else 'ok',
            'expired': 0, 'retried': 0}
        assert prior._audit(conn, lease) == before
        assert _cache(conn, lease) == cache
        if generation is not None:
            prior.restart._abort(conn, generation)
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason=''")
        conn.commit()


def test_rollback_then_duplicate_recovery_and_claim_is_one_retry(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease, _ = _seed(conn)
        before = prior._audit(conn, lease)
        assert prior._expire(conn, commit=False)['expired'] == 1
        conn.rollback()
        assert prior._audit(conn, lease) == before
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('DELETE FROM public.lab_arena_runs WHERE assignment_id<>%s', (lease['assignment_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
    def expire(_):
        with psycopg.connect(**dsn) as conn:
            return prior._expire(conn)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(expire, range(2)))
    assert sorted(x['expired'] for x in results) == [0, 1]
    def claim(suffix):
        with psycopg.connect(**dsn) as conn:
            return prior.recovery._claim(conn, prior.recovery.prior.RUNNER_B, suffix)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, ['b', 'c']))
    assert sum(x['status'] == 'leased' for x in claims) == 1
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=2', (lease['assignment_id'],))
        assert cur.fetchone()[0] == 1


@pytest.mark.parametrize('winner', ['reservation', 'dispatch', 'recovery'])
def test_unpriced_recovery_fences_queued_paid_work(database, migrated, winner):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease, _ = _seed(conn)
        before = prior._audit(conn, lease)
    owner = psycopg.connect(**dsn)
    try:
        if winner == 'reservation':
            assert prior._reserve(owner, lease, prior.SECOND_CALL, commit=False)['status'] == 'reserved'
        elif winner == 'dispatch':
            assert prior._reserve(owner, lease, prior.SECOND_CALL)['status'] == 'reserved'
            assert prior._dispatch(owner, lease, prior.SECOND_CALL, commit=False)['status'] == 'dispatched'
        else:
            assert prior._expire(owner, commit=False)['expired'] == 1
        started = threading.Event()
        def losing_call():
            with psycopg.connect(**dsn) as conn:
                started.set()
                return (prior._expire(conn) if winner != 'recovery'
                        else prior._reserve(conn, lease, prior.SECOND_CALL))
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
                    pytest.fail('completed-response recovery did not reach the DB lock')
            owner.commit()
            answer = future.result(timeout=5)
        if winner == 'recovery':
            assert answer['status'] == 'stale'
            assert prior._dispatch(owner, lease, prior.SECOND_CALL)['status'] == 'stale'
            assert prior._audit(owner, lease)['lab_arena_ledger'] == before['lab_arena_ledger']
        else:
            assert answer['expired'] == 0
        with owner.cursor() as cur:
            cur.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='uncertain'", (lease['run_id'],))
            assert cur.fetchone()[0] == 1
    finally:
        owner.close()


# Reuse the prior production-path controls on the upgraded function, including
# real reservation/dispatch/expiry races, zero-call and paid execute behavior.
test_prior_settled = prior.test_last_settlement_after_abandonment_releases_once_with_renewed_lease
test_prior_nonsettlement = prior.test_any_nonsettlement_head_blocks_even_with_other_paid_settlements
test_prior_authenticated_cleanup = prior.test_only_exact_terminated_current_lease_recovers
test_prior_zero_call = prior.test_zero_call_412_recovery_is_unchanged
test_prior_inflight_races = prior.test_queued_provider_work_cannot_cross_recovery_fence
test_prior_execute_cost = prior.test_execute_retry_retains_spend_and_enters_normal_scoring
test_prior_execute_cutoff = prior.test_execute_recovery_does_not_extend_the_frozen_stage_cutoff
test_prior_execute_cap = prior.test_execute_retry_cannot_reset_confirmed_spend_at_the_icp_cap
