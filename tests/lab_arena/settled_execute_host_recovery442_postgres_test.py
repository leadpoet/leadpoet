"""Settled execute abandonment retains ordinary retries, costs and lease fences."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

import pytest

from tests.lab_arena import abandoned_host_recovery412_postgres_test as recovery
from tests.lab_arena import abandoned_host_restart_guard414_postgres_test as restart
from tests.lab_arena import settled_score_host_recovery441_postgres_test as score
from tests.lab_arena import score_host_retry398_postgres_test as identity

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/442-lab-arena-settled-execute-host-recovery.sql'
PREHASH = '9735de28e3a59f9143546b3a12d5345fc1627ea8d9b39a80f8863177286ff654'
database = recovery.database
TOKEN_HASH = 'sha256:' + 'a' * 64
CALL = 'sha256:' + 'e' * 64
SECOND_CALL = 'sha256:' + 'f' * 64


@pytest.fixture(scope='module')
def migrated(database):
    # Install the production upgrade chain, not a handwritten expiry function.
    score.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            before = identity._security(cur)
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (identity.EXPIRY,))
            old_definition = cur.fetchone()[0]
            claim_hash = identity._hash(cur, 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)')
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
            assert identity._hash(cur, 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)') == claim_hash
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (identity.EXPIRY,))
            new_definition = cur.fetchone()[0]
            # Only the paid branch changes. Ordinary expiry, zero-call recovery
            # and the claim quarantine remain the exact production definitions.
            assert old_definition.split('  -- lab_arena_settled_score_host_recovery_v1:')[0] == new_definition.split('  -- lab_arena_settled_execute_host_recovery_v1:')[0]
            tail = '  FOR v_run IN\n    SELECT * FROM public.lab_arena_runs\n    WHERE round_id = p_round_id AND status = \'leased\''
            assert old_definition.split(tail)[1] == new_definition.split(tail)[1]
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


def _settle(conn, lease, call=CALL, *, commit=True, token_hash=TOKEN_HASH, amount=100):
    return _rpc(conn, "lab_arena_settle_call(%s,%s,%s,%s,%s::jsonb,4500)",
                (lease['run_id'], token_hash, call, amount,
                 json.dumps({'status': 200, 'call_succeeded': True, 'call': {'call_succeeded': True}})), commit=commit)


def _expire(conn, *, commit=True):
    return _rpc(conn, 'lab_arena_expire_leases(%s)', (recovery.prior.ROUND,), commit=commit)


def _seed(conn, *, abandoned=True, kind='score', settled=True, amount=100):
    lease = recovery._seed(conn, kind=kind, event=False)
    _event(conn, lease, 'runtime.started', {'status': 'starting', 'runtime': 'runsc',
                                         'lease_generation': lease['lease_generation']})
    assert _reserve(conn, lease)['status'] == 'reserved'
    assert _dispatch(conn, lease)['status'] == 'dispatched'
    if settled:
        assert _settle(conn, lease, amount=amount)['status'] == 'settled'
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


@pytest.mark.parametrize('kind', ['score', 'execute'])
def test_last_settlement_after_abandonment_releases_once_with_renewed_lease(database, migrated, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, settled=False, kind=kind)
        if kind == 'score':
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
        assert doc['recovery_reason'] == f'authenticated_settled_{kind}_runtime_host_error'
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


@pytest.mark.parametrize('kind', ['score', 'execute'])
@pytest.mark.parametrize('head', ['reservation', 'dispatch', 'uncertain', 'recovery', 'refusal'])
def test_any_nonsettlement_head_blocks_even_with_other_paid_settlements(database, migrated, head, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, kind=kind)
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


@pytest.mark.parametrize('kind', ['score', 'execute'])
@pytest.mark.parametrize('change', [
    'active', 'wrong_runner', 'wrong_round', 'wrong_submission',
    'wrong_miner', 'wrong_assignment', 'wrong_stage', 'wrong_position',
    'wrong_attempt', 'wrong_kind', 'wrong_class', 'wrong_status', 'setup',
    'cleanup', 'secondary_cleanup', 'finished', 'result', 'output',
    'old_stage_generation', 'wrong_lease_generation', 'wrong_claim_generation',
    'no_started', 'wrong_started_generation', 'wrong_started_runner',
    'before_claim', 'after_claim_expiry', 'non_signaled', 'positive_exit',
    'string_exit', 'timed_out', 'missing_timeout', 'null_identity',
    'operator_pause', 'restart_guard',
])
def test_only_exact_terminated_current_lease_recovers(database, migrated, change, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, abandoned=change != 'active',
                      kind=kind)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            columns = {'wrong_runner': ('runner_hotkey', recovery.prior.RUNNER_B),
                       'wrong_round': ('round_id', 'other'), 'wrong_submission': ('submission_id', 'other'),
                       'wrong_miner': ('miner_hotkey', recovery.prior.RUNNER_B),
                       'wrong_assignment': ('assignment_id', 'other'), 'wrong_stage': ('stage', 2),
                       'wrong_position': ('icp_position', 9), 'wrong_attempt': ('attempt', 2),
                       'wrong_kind': ('run_kind', 'execute' if kind == 'score' else 'score')}
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


@pytest.mark.parametrize('kind', ['score', 'execute'])
@pytest.mark.parametrize('winner', ['reservation', 'dispatch', 'recovery'])
def test_queued_provider_work_cannot_cross_recovery_fence(database, migrated, winner, kind):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, kind=kind)
    owner = psycopg.connect(**dsn)
    try:
        if winner == 'reservation':
            assert _reserve(owner, lease, SECOND_CALL, commit=False)['status'] == 'reserved'
        elif winner == 'dispatch':
            assert _reserve(owner, lease, SECOND_CALL)['status'] == 'reserved'
            assert _dispatch(owner, lease, SECOND_CALL, commit=False)['status'] == 'dispatched'
        else:
            assert _expire(owner, commit=False)['expired'] == 1
        started = threading.Event()

        def losing_call():
            with psycopg.connect(**dsn) as loser:
                started.set()
                return _expire(loser) if winner != 'recovery' else _reserve(loser, lease, SECOND_CALL)

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
        assert answer['expired'] == 0 if winner != 'recovery' else answer['status'] == 'stale'
        # A delayed dispatch uses the same lease fence and cannot spend after recovery.
        if winner == 'recovery':
            assert _dispatch(owner, lease)['status'] == 'stale'
        with owner.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone()[0] == {'reservation': 4, 'dispatch': 5, 'recovery': 3}[winner]
    finally:
        owner.close()


def test_execute_retry_retains_spend_and_enters_normal_scoring(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, kind='execute')
        with conn.cursor() as cur:
            # Configure this isolated fixture with the production cost policy.
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
            cur.execute('DELETE FROM public.lab_arena_runs WHERE assignment_id<>%s',
                        (lease['assignment_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        before = _audit(conn, lease)
        cost_args = (recovery.prior.ROUND, lease['submission_id'], lease['icp_position'], 1)
        cost = _rpc(conn, 'lab_arena_icp_cost_eligibility(%s,%s,%s,%s)', cost_args)
        assert cost['competition_sourcing_microusd'] == 100 and cost['eligible']
        assert _expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 1}
        assert _audit(conn, lease)['lab_arena_ledger'] == before['lab_arena_ledger']
        retry = recovery._claim(conn, recovery.prior.RUNNER_B, 'c')
        assert (retry['status'], retry['kind'], retry['attempt']) == ('leased', 'execute', 2)
        token = 'sha256:' + 'c' * 64
        reserved = _reserve(conn, retry, SECOND_CALL, token_hash=token)
        assert reserved['status'] == 'reserved'
        # Migration 321 uses confirmed spend and zero monetary holds.
        assert reserved['amount_microusd'] == 0
        assert _dispatch(conn, retry, SECOND_CALL, token_hash=token)['status'] == 'dispatched'
        assert _settle(conn, retry, SECOND_CALL, token_hash=token)['status'] == 'settled'
        cost = _rpc(conn, 'lab_arena_icp_cost_eligibility(%s,%s,%s,%s)', cost_args)
        assert cost['competition_sourcing_microusd'] == 200
        assert cost['execution']['successful_calls'] == 2 and cost['eligible']
        output_ref = 'arena/fixture/retry-output.json'
        completed = _rpc(conn, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
                         (retry['run_id'], token, json.dumps({'terminal_status': 'accepted'}),
                          'accepted', output_ref))
        assert completed['status'] == 'accepted'
        with conn.cursor() as cur:
            cur.execute('SELECT configuration_doc,arena_network_name,arena_netuid,evaluation_date '
                        'FROM public.lab_arena_rounds WHERE round_id=%s', (recovery.prior.ROUND,))
            config, network, netuid, evaluation_date = cur.fetchone()
            cache_key, input_hash = 'sha256:' + '1' * 64, 'sha256:' + '2' * 64
            scope = {'cache_key': cache_key, 'scoring_input_hash': input_hash,
                     'round_id': recovery.prior.ROUND, 'network_name': network, 'netuid': netuid,
                     'integrity_policy': 'arena_integrity_v1', 'evaluation_date': evaluation_date,
                     'scorer_image_digest': config['scorer_image_digest'],
                     'scorer_image_reference': config['scorer_image_reference']}
            work = {'scored_run_id': retry['run_id'], 'submission_id': retry['submission_id'],
                    'icp_position': retry['icp_position'], 'output_ref': output_ref,
                    'judgment_cache_key': cache_key, 'judgment_input_hash': input_hash,
                    'judgment_scope_doc': scope, 'judgment_group_leader': True,
                    'judgment_group_miner_hotkeys': [retry['miner_hotkey']]}
            # The driver has committed this plan before calling the real RPC.
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage1_closed',"
                        'stage1_scoring_plan_doc=%s::jsonb', (json.dumps({'work_items': [work]}),))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        opened = _rpc(conn, 'lab_arena_open_scoring_v2(%s,1::smallint,%s::jsonb)',
                      (recovery.prior.ROUND, json.dumps([work])))
        assert opened['status'] == 'ok' and opened['assignments'] == 1
        score_lease = recovery._claim(conn, recovery.prior.RUNNER_A, 'd')
        assert score_lease['status'] == 'leased' and score_lease['kind'] == 'score'
        assert score_lease['scored_run_id'] == retry['run_id']
        assert _audit(conn, lease)['lab_arena_ledger'] == before['lab_arena_ledger']


def test_execute_recovery_does_not_extend_the_frozen_stage_cutoff(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, kind='execute')
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                        "jsonb_set(configuration_doc || "
                        "'{\"execution_sequence_policy\":\"baseline_scored_first_v1\"}'::jsonb,"
                        "'{schedule,stage_1_close}',to_jsonb((now()-interval '1 second')::text))")
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert _expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 0}
        with conn.cursor() as cur:
            cur.execute('SELECT max(attempt) FROM public.lab_arena_runs WHERE assignment_id=%s',
                        (lease['assignment_id'],))
            assert cur.fetchone()[0] == 1


def test_execute_retry_cannot_reset_confirmed_spend_at_the_icp_cap(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, kind='execute', amount=4000000)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                        "configuration_doc || %s::jsonb", (json.dumps({
                            'sourcing_cost_eligibility_policy': 'successful_calls_per_icp_v1',
                            'integrity_policy': 'arena_integrity_v1',
                            'execution_icp_cap_microusd': 4000000,
                            'cost_per_company_microusd': 800000}),))
            cur.execute('DELETE FROM public.lab_arena_runs WHERE assignment_id<>%s',
                        (lease['assignment_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        before = _audit(conn, lease)
        assert _expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 1}
        retry = recovery._claim(conn, recovery.prior.RUNNER_B, 'c')
        assert retry['status'] == 'leased' and retry['attempt'] == 2
        refusal = _reserve(conn, retry, SECOND_CALL, token_hash='sha256:' + 'c' * 64)
        assert refusal['status'] == 'refused' and refusal['reason'] == 'money_cap'
        cost = _rpc(conn, 'lab_arena_icp_cost_eligibility(%s,%s,%s,%s)',
                    (recovery.prior.ROUND, lease['submission_id'], lease['icp_position'], 5))
        assert cost['competition_sourcing_microusd'] == 4000000
        assert _audit(conn, lease)['lab_arena_ledger'] == before['lab_arena_ledger']
