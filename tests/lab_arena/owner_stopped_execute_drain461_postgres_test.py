"""Guard owners can drain only authenticated, fully settled stopped execute work."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena import owner_stopped_score_drain454_postgres_test as prior

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/461-lab-arena-owner-authorized-stopped-execute-drain.sql'
PREHASH = 'dadb3e139bb972ca8d3ef4894e0c2e3ba087f55c57973adaa21ed1a8d46d6c0a'
database = prior.database
fresh_control = prior.fresh_control
paid = prior.prior.prior
guard = prior.guard


def _reset(conn):
    with conn.cursor() as cur:
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,"
                    "pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',"
                    "guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',"
                    "captured_leases='[]'")
    conn.commit()


def _seed(conn, *, settled=1, uncertain=0, abandoned=True, signal=-2):
    lease = paid._seed(conn, kind='execute', abandoned=False)
    for index in range(1, settled + uncertain):
        call = 'sha256:' + hashlib.sha256(f'execute-drain-{index}'.encode()).hexdigest()
        reservation = (paid._reserve(conn, lease, call) if index < settled else
            paid._rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'exa.contents','deepline','host',100,'{}'::jsonb,4500)",
                      (lease['run_id'], paid.TOKEN_HASH, call)))
        assert reservation['status'] == 'reserved', reservation
        assert paid._dispatch(conn, lease, call)['status'] == 'dispatched'
        if index < settled:
            assert paid._settle(conn, lease, call)['status'] == 'settled'
        else:
            assert paid._rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
                (lease['run_id'], paid.TOKEN_HASH, call,
                 json.dumps({'call_succeeded': False, 'provider_status': 500})))['status'] == 'uncertain'
    if abandoned:
        paid._event(conn, lease, 'runtime.error', {
            'status': 'abandoned', 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError',
            'runtime_host_reason': 'sandbox_launcher_signaled',
            'launch_exit_code': signal, 'launch_timed_out': False,
        })
    return lease


def _round(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',
                    (paid.recovery.prior.ROUND,))
        return cur.fetchone()[0]


@pytest.fixture(scope='module')
def migrated(database):
    # Exercise the real production upgrade chain through454 before461.
    prior.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        _reset(conn)
        lease = _seed(conn)
        generation = prior._hold_capture(conn)
        waiting = guard._quiescence(conn, generation=generation)
        assert (waiting['still_leased_count'], waiting['expired_receipt_count']) == (1, 0)
        with conn.cursor() as cur:
            before = prior._state(cur)
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREHASH, '0' * 64, 1))
            cur.execute('ROLLBACK')
            assert prior._state(cur) == before
            cur.execute(SQL.read_text())
            after = prior._state(cur)
            assert [row[2:] for row in after] == [row[2:] for row in before]
            old = {row[0]: row[1] for row in before}
            new = {row[0]: row[1] for row in after}
            expiry = 'lab_arena_expire_leases(text)'
            paid_boundary = '  -- lab_arena_settled_execute_host_recovery_v1:'
            # The entire existing stopped-host proof, ledger rules, retry bounds,
            # and ordinary expiry are byte-identical after the candidate scan.
            proof = "    IF v_run.status = 'leased'"
            assert old[expiry].split(paid_boundary)[1].split(proof, 1)[1] == new[expiry].split(paid_boundary)[1].split(proof, 1)[1]
            zero = '  -- The existing whole-round expiry must never touch foreign captured work.'
            assert old[expiry].split(zero)[0] == new[expiry].split(zero)[0]
            cur.execute(SQL.read_text())
            assert prior._state(cur) == after
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['expired_receipt_count'], drained['preserved']) == (1, True)
    return True


@pytest.mark.parametrize('count,signal', [(1, -9), (87, -2), (150, -9)])
def test_captured_settled_execute_preserves_evidence_and_retries_once(database, migrated, count, signal):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, settled=count, signal=signal)
        generation = prior._hold_capture(conn)
        before, frozen = paid._audit(conn, lease), _round(conn)
        assert paid._expire(conn)['expired'] == 0  # Unowned expiry stays held.
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['captured_count'], drained['expired_receipt_count'],
                drained['lost_or_mutated_count'], drained['pending_retry_count'],
                drained['preserved']) == (1, 1, 0, 1, True)
        assert guard._quiescence(conn, generation=generation)['outcome_commitment'] == drained['outcome_commitment']
        after = paid._audit(conn, lease)
        assert after['lab_arena_ledger'] == before['lab_arena_ledger']
        assert after['lab_arena_trajectory_events'] == before['lab_arena_trajectory_events']
        assert after['run'][0:2] == ('failed', 'lease_expired')
        assert after['run'][2]['recovery_reason'] == 'authenticated_settled_execute_runtime_host_error'
        with conn.cursor() as cur:
            cur.execute('SELECT %s::timestamptz', (after['run'][2]['original_lease_expires_at'],))
            assert cur.fetchone()[0] == before['run'][4]
        assert after['run'][3] == lease
        assert _round(conn) == frozen
        assert paid.recovery._claim(conn, paid.recovery.prior.RUNNER_B, 'b')['status'] == 'paused'
        assert guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1',
            guard.GUARD, guard.OWNER, generation, 'gateway_destructive')['restart_phase'] == 'gateway_destructive'


@pytest.mark.parametrize('fault', [
    'owner', 'generation', 'context', 'expired_guard', 'uncaptured', 'capture_generation', 'claim_generation',
    'nohost', 'cleanup', 'wrong_runner', 'output', 'result', 'dispatch', 'uncertain',
])
def test_execute_recovery_keeps_owner_lease_cleanup_output_and_settlement_gates(database, migrated, fault):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = (_seed(conn, settled=111, uncertain=2) if fault == 'uncertain'
                 else _seed(conn, abandoned=fault != 'nohost'))
        if fault == 'dispatch':
            assert paid._reserve(conn, lease, paid.SECOND_CALL)['status'] == 'reserved'
            assert paid._dispatch(conn, lease, paid.SECOND_CALL)['status'] == 'dispatched'
        elif fault == 'cleanup':
            paid._event(conn, lease, 'runtime.cleanup_error', {})
        generation = prior._hold_capture(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault == 'uncaptured':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases='[]'")
            elif fault == 'expired_guard':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET guard_expires_at=now()-interval '1 second'")
            elif fault == 'capture_generation':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases=jsonb_set(captured_leases,'{0,lease_generation}','999')")
            elif fault == 'claim_generation':
                cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{lease_generation}','999') WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'wrong_runner':
                cur.execute('UPDATE public.lab_arena_runs SET runner_hotkey=%s WHERE run_id=%s',
                            (paid.recovery.prior.RUNNER_B, lease['run_id']))
            elif fault == 'output':
                cur.execute("UPDATE public.lab_arena_runs SET output_ref='arena/retained/output.json' WHERE run_id=%s", (lease['run_id'],))
            elif fault == 'result':
                cur.execute("UPDATE public.lab_arena_runs SET result_doc='{}' WHERE run_id=%s", (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        before = paid._audit(conn, lease)
        if fault in ('owner', 'generation'):
            with pytest.raises(psycopg.Error, match='owner_or_generation_differs'):
                guard._quiescence(conn, owner='lab_arena_restart_owner:'+'f'*64 if fault == 'owner' else guard.OWNER,
                                  generation=generation+1 if fault == 'generation' else generation)
            conn.rollback()
        elif fault in ('output', 'result'):
            with pytest.raises(psycopg.Error, match='captured write set differs'):
                guard._quiescence(conn, generation=generation)
            conn.rollback()
        elif fault == 'context':
            with conn.cursor() as cur:
                cur.execute('SELECT set_config(%s,%s,TRUE)', (prior.CONTEXT, json.dumps({
                    'guard_id': guard.GUARD, 'owner_id': 'wrong', 'guard_generation': generation,
                })))
            assert paid._expire(conn)['expired'] == 0
        else:
            drained = guard._quiescence(conn, generation=generation)
            assert drained['expired_receipt_count'] == 0 and not drained['preserved']
        assert paid._audit(conn, lease) == before


def test_mixed_captured_live_execute_stays_untouched_while_stopped_execute_expires(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        stopped = _seed(conn, abandoned=False)
        live = paid.recovery._claim(conn, paid.recovery.prior.RUNNER_A, 'b')
        assert live['status'] == 'leased'
        paid._event(conn, stopped, 'runtime.error', {
            'status': 'abandoned', 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError',
            'runtime_host_reason': 'sandbox_launcher_signaled',
            'launch_exit_code': -9, 'launch_timed_out': False,
        })
        generation = prior._hold_capture(conn)
        before_live, before_dead = paid._audit(conn, live), paid._audit(conn, stopped)
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['captured_count'], drained['still_leased_count'],
                drained['expired_receipt_count'], drained['lost_or_mutated_count'],
                drained['pending_retry_count'], drained['preserved']) == (2, 1, 1, 0, 1, False)
        assert guard._quiescence(conn, generation=generation)['outcome_commitment'] == drained['outcome_commitment']
        assert paid._audit(conn, live) == before_live
        after_dead = paid._audit(conn, stopped)
        assert after_dead['lab_arena_ledger'] == before_dead['lab_arena_ledger']
        assert after_dead['lab_arena_trajectory_events'] == before_dead['lab_arena_trajectory_events']


@pytest.mark.parametrize('unpriced', [False, True])
def test_score_settled_and_completed_unpriced_controls_still_drain(database, migrated, unpriced):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease, _ = prior.prior._seed(conn, count=int(unpriced))
        generation = prior._hold_capture(conn)
        before, cache = paid._audit(conn, lease), prior.prior._cache(conn, lease)
        drained = guard._quiescence(conn, generation=generation)
        assert drained['preserved'] and drained['expired_receipt_count'] == 1
        after = paid._audit(conn, lease)
        assert after['lab_arena_ledger'] == before['lab_arena_ledger']
        assert prior.prior._cache(conn, lease) == cache
        expected = ('authenticated_completed_unpriced_score_runtime_host_error' if unpriced
                    else 'authenticated_settled_score_runtime_host_error')
        assert after['run'][2]['recovery_reason'] == expected


def test_completed_unpriced_execute_is_not_a_settled_execute(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease, _ = prior.prior._seed(conn, kind='execute', count=1)
        generation = prior._hold_capture(conn)
        before = paid._audit(conn, lease)
        drained = guard._quiescence(conn, generation=generation)
        assert not drained['preserved'] and drained['expired_receipt_count'] == 0
        assert paid._audit(conn, lease) == before


def test_uncertain_execute_natural_expiry_preserves_bills_and_creates_one_retry(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = _seed(conn, settled=111, uncertain=2)
        with conn.cursor() as cur:
            # Move the disposable claim and its authenticated runtime window
            # together. Keep the effective expiry one second ahead, then let
            # real database time cross it without changing the captured claim.
            cur.execute('SELECT lease_expires_at-(clock_timestamp()+interval \'1 second\') '
                        'FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            shift = cur.fetchone()[0]
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=lease_expires_at-%s,"
                        "claim_response=jsonb_set(claim_response,'{lease_expires_at}',"
                        "to_jsonb(((claim_response->>'lease_expires_at')::timestamptz-%s)::text)) "
                        "WHERE run_id=%s", (shift, shift, lease['run_id']))
            cur.execute('UPDATE public.lab_arena_trajectory_events '
                        'SET created_at=created_at-%s,occurred_at=occurred_at-%s '
                        'WHERE run_id=%s', (shift, shift, lease['run_id']))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        generation = prior._hold_capture(conn)
        before, frozen = paid._audit(conn, lease), _round(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT entry_kind,count(*) FROM (SELECT DISTINCT ON (call_identity) "
                        "entry_kind FROM public.lab_arena_ledger WHERE run_id=%s "
                        "ORDER BY call_identity,entry_id DESC) heads GROUP BY entry_kind",
                        (lease['run_id'],))
            assert dict(cur.fetchall()) == {'settlement': 111, 'uncertain': 2}
            cur.execute('SELECT public.lab_arena__submission_kind_spend(%s,\'execute\')',
                        (lease['submission_id'],))
            spend = cur.fetchone()[0]
        waiting = guard._quiescence(conn, generation=generation)
        assert (waiting['still_leased_count'], waiting['expired_receipt_count'],
                waiting['pending_retry_count'], waiting['preserved']) == (1, 0, 0, False)
        assert paid._audit(conn, lease) == before
        with conn.cursor() as cur:
            cur.execute('SELECT pg_sleep(greatest(0,extract(epoch FROM '
                        '(lease_expires_at-clock_timestamp())))+0.02) '
                        'FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['captured_count'], drained['still_leased_count'],
                drained['expired_receipt_count'], drained['lost_or_mutated_count'],
                drained['pending_retry_count'], drained['preserved']) == (1, 0, 1, 0, 1, True)
        assert guard._quiescence(conn, generation=generation)['outcome_commitment'] == drained['outcome_commitment']
        after = paid._audit(conn, lease)
        assert after['lab_arena_ledger'] == before['lab_arena_ledger']
        assert after['lab_arena_trajectory_events'] == before['lab_arena_trajectory_events']
        assert after['run'][0:2] == ('failed', 'lease_expired')
        assert set(after['run'][2]) == {'expired_at'}  # Normal expiry receipt.
        assert after['run'][3:] == before['run'][3:]
        assert _round(conn) == frozen
        with conn.cursor() as cur:
            cur.execute('SELECT status,attempt FROM public.lab_arena_runs '
                        'WHERE assignment_id=%s ORDER BY attempt', (lease['assignment_id'],))
            assert cur.fetchall() == [('failed', 1), ('pending', 2)]
            cur.execute('SELECT public.lab_arena__submission_kind_spend(%s,\'execute\')',
                        (lease['submission_id'],))
            assert cur.fetchone()[0] == spend
        assert paid.recovery._claim(conn, paid.recovery.prior.RUNNER_B, 'b')['status'] == 'paused'
        assert guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1',
            guard.GUARD, guard.OWNER, generation, 'gateway_destructive')['restart_phase'] == 'gateway_destructive'
