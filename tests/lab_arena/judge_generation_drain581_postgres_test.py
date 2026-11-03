"""A canonical drain preserves judges already admitted before stage advancement."""
from pathlib import Path

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena import score_host_continuation575_postgres_test as prior
from tests.lab_arena import restart_expired_zero_call394_postgres_test as guard
from tests.lab_arena import settled_score_host_recovery441_postgres_test as paid
from tests.lab_arena.quiescence_recovery_gate522_postgres_test import HOST_ERROR

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/581-lab-arena-judge-generation-drain.sql'
scenario = prior.scenario
SIGNATURES = (prior.SIG, 'public.lab_arena__restart_drain_state_v1()',
              'public.lab_arena_restart_quiescence_v1(text,text,bigint)')
AUTHORITY = ('public.lab_arena__judge_continuation_v1('
             'public.lab_arena_rounds,public.lab_arena_runs)')


@pytest.fixture(autouse=True)
def restore(scenario):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(oid) FROM pg_proc '
                    'WHERE oid=ANY(%s::regprocedure[])', (list(SIGNATURES),))
        sources = [row[0] for row in cur.fetchall()]
        cur.execute('TRUNCATE public.lab_arena_trajectory_events CASCADE')
        cur.execute('TRUNCATE public.lab_arena_company_judgment_reservations CASCADE')
        cur.execute(prior.SQL.read_text())
    yield
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        for source in sources:
            cur.execute(source)
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,"
                    "pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',"
                    "guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")


def capture(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control')
        generation = cur.fetchone()[0]
    result = guard._acquire(conn, generation)
    conn.commit()
    return result['guard_generation']


def test_admitted_older_judge_can_drain_after_real_stage_transition(scenario):
    db = scenario[0]
    lease = prior.prepare(scenario, error=False)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        token = hash_lease_token(lease['lease_token'])
        paid._event(conn, lease, 'runtime.started', {'status': 'starting', 'runtime': 'runsc',
            'lease_generation': lease['lease_generation']}, token_hash=token)
        assert paid._reserve(conn, lease, token_hash=token)['status'] == 'reserved'
        assert paid._dispatch(conn, lease, token_hash=token)['status'] == 'dispatched'
        assert paid._settle(conn, lease, token_hash=token)['status'] == 'settled'
        paid._event(conn, lease, 'runtime.error', HOST_ERROR, token_hash=token)
        before = prior.state(cur, lease)
        assert prior.prior.scalar(cur, 'public.lab_arena__judge_continuation_v1(d,r) '
            'FROM public.lab_arena_rounds d JOIN public.lab_arena_runs r USING(round_id) '
            'WHERE r.run_id=%s', (lease['run_id'],)) is True
        generation = capture(conn)
        with pytest.raises(db[0].Error, match='captured write set differs'):
            guard._quiescence(conn, generation=generation)
        conn.rollback()
        frozen = prior.prior.prior.fingerprint(cur, lease['round_id'])
        cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id')
        ledger = cur.fetchall()
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        conn.commit()
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['preserved'], drained['lost_or_mutated_count'],
                drained['expired_receipt_count'], drained['pending_retry_count']) == (True, 0, 1, 1)
        assert prior.state(cur, lease)['claim_response'] == before['claim_response']
        assert prior.prior.prior.fingerprint(cur, lease['round_id']) == frozen
        cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id')
        assert cur.fetchall() == ledger
        assert guard._quiescence(conn, generation=generation)['outcome_commitment'] == drained['outcome_commitment']
        guard._rpc(conn, 'lab_arena_authorize_restart_phase_v1', guard.GUARD,
                   guard.OWNER, generation, 'gateway_destructive')
        guard._rpc(conn, 'lab_arena_mark_restart_ready_v1', guard.GUARD,
                   guard.OWNER, generation, 'gateway_ready')
        assert not guard._rpc(conn, 'lab_arena_release_restart_guard_v1', guard.GUARD,
                              guard.OWNER, generation, 'test-stage-drain')['guard_present']
        # Move only the local fixture's backoff clock. Production backoff is
        # unchanged; the normal runner must accept the preserved retry.
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '30 minutes' WHERE assignment_id=%s AND status='pending'", (lease['assignment_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
    h = scenario[1]
    h.advance_until('scored', runners=1)
    assert h.service.store.get_run(lease['assignment_id'] + ':2')['status'] == 'accepted'
    assert h.service.publish(lease['round_id'])['status'] == 'ok'
    assert h.status() == 'published'


def paid_lease(scenario, *, fault=None):
    lease = prior.prepare(scenario, error=False)
    db = scenario[0]
    token = hash_lease_token(lease['lease_token'])
    with db[0].connect(**db[1]) as conn:
        if fault != 'no_start':
            paid._event(conn, lease, 'runtime.started', {'status': 'starting',
                'lease_generation': lease['lease_generation']}, token_hash=token)
        assert paid._reserve(conn, lease, token_hash=token)['status'] == 'reserved'
        if fault != 'reservation':
            assert paid._dispatch(conn, lease, token_hash=token)['status'] == 'dispatched'
        if fault not in ('reservation', 'dispatch'):
            assert paid._settle(conn, lease, token_hash=token)['status'] == 'settled'
        if fault != 'healthy':
            error = dict(HOST_ERROR)
            if fault == 'wrong_error':
                error['error_class'] = 'UntrustedHostError'
            paid._event(conn, lease, 'runtime.error', error, token_hash=token)
        if fault == 'cleanup':
            paid._event(conn, lease, 'runtime.cleanup_error', {'status': 'error'}, token_hash=token)
    return lease


@pytest.mark.parametrize('fault', ['healthy', 'no_start', 'wrong_error', 'cleanup', 'reservation', 'dispatch'])
def test_older_judge_still_requires_stopped_host_and_settled_calls(scenario, fault):
    db = scenario[0]
    lease = paid_lease(scenario, fault=fault)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
        generation = capture(conn)
        before = paid._audit(conn, lease)
        outcome = guard._quiescence(conn, generation=generation)
        assert (outcome['still_leased_count'], outcome['expired_receipt_count'],
                outcome['lost_or_mutated_count']) == (1, 0, 0)
        assert paid._audit(conn, lease) == before


@pytest.mark.parametrize('fault', ['floor', 'source', 'future_generation', 'capture', 'owner'])
def test_mutated_authority_never_receives_restart_permission(scenario, fault):
    db = scenario[0]
    lease = paid_lease(scenario)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
        generation = capture(conn)
        cur.execute('SET LOCAL session_replication_role=replica')
        if fault == 'floor':
            cur.execute('UPDATE public.lab_arena_runs SET scoring_generation_floor=scoring_generation_floor+1 WHERE run_id=%s', (lease['run_id'],))
        elif fault == 'source':
            cur.execute("UPDATE public.lab_arena_runs SET status='pending' WHERE run_id=(SELECT scored_run_id FROM public.lab_arena_runs WHERE run_id=%s)", (lease['run_id'],))
        elif fault == 'future_generation':
            cur.execute('UPDATE public.lab_arena_runs SET stage_generation=(SELECT stage_generation+1 FROM public.lab_arena_rounds WHERE round_id=%s) WHERE run_id=%s', (lease['round_id'], lease['run_id']))
        elif fault == 'capture':
            cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases='[]'")
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        before = paid._audit(conn, lease)
        if fault != 'capture':
            args = {'generation': generation}
            if fault == 'owner':
                args['owner'] = 'lab_arena_restart_owner:' + 'f' * 64
            with pytest.raises(db[0].Error):
                guard._quiescence(conn, **args)
            conn.rollback()
        else:
            outcome = guard._quiescence(conn, generation=generation)
            assert not outcome['preserved'] and outcome['current_leased_count'] == 1
        assert paid._audit(conn, lease) == before


@pytest.mark.parametrize('accounting', ['none', 'reservation', 'dispatch'])
def test_natural_expiry_preserves_old_judge_liability_and_retry(scenario, accounting):
    db = scenario[0]
    lease = prior.prepare(scenario, error=False)
    token = hash_lease_token(lease['lease_token'])
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        if accounting != 'none':
            assert paid._reserve(conn, lease, token_hash=token)['status'] == 'reserved'
        if accounting == 'dispatch':
            assert paid._dispatch(conn, lease, token_hash=token)['status'] == 'dispatched'
        cur.execute(SQL.read_text())
        generation = capture(conn)
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease['run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        before = paid._audit(conn, lease)
        outcome = guard._quiescence(conn, generation=generation)
        after = paid._audit(conn, lease)
        assert outcome['preserved'] and outcome['expired_receipt_count'] == 1
        assert outcome['lost_or_mutated_count'] == 0
        assert before['lab_arena_trajectory_events'] == after['lab_arena_trajectory_events']
        assert before['run'][3] == after['run'][3]
        if accounting == 'none':
            assert before['lab_arena_ledger'] == after['lab_arena_ledger'] == []
        else:
            assert after['lab_arena_ledger'][:-1] == before['lab_arena_ledger']
            assert after['lab_arena_ledger'][-1]['entry_kind'] == ('uncertain' if accounting == 'dispatch' else 'recovery')
        # Natural expiry retains the ordinary dormant retry and any uncertain
        # liability; the guarded restart must not dispatch or forgive it.
        assert outcome['pending_retry_count'] == 1
        assert guard._quiescence(conn, generation=generation)['outcome_commitment'] == outcome['outcome_commitment']


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('signature', (*SIGNATURES, AUTHORITY))
@pytest.mark.parametrize('drift', ['source', 'acl', 'security'])
def test_migration_is_atomic_and_rejects_function_or_authority_drift(scenario, installed, signature, drift):
    db = scenario[0]
    body = SQL.read_text().split('DO $judge_generation_drain581$')[1].split('$judge_generation_drain581$;')[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        if installed:
            cur.execute(SQL.read_text())
        cur.execute('BEGIN')
        cur.execute('SAVEPOINT hostile')
        if drift == 'source':
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
            cur.execute(cur.fetchone()[0].replace('AS $function$', 'AS $function$\n-- unknown drift', 1))
        elif drift == 'acl':
            cur.execute('GRANT EXECUTE ON FUNCTION ' + signature + ' TO anon')
        else:
            cur.execute('ALTER FUNCTION ' + signature + ' SECURITY INVOKER')
        with pytest.raises(db[0].Error, match='judge generation drain'):
            cur.execute('DO $test$' + body + '$test$')
        cur.execute('ROLLBACK TO SAVEPOINT hostile')


def test_complete_round_preserves_baseline_miners_and_publication(scenario):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(SQL.read_text())
    prior.prior.test_real_overlap_and_final_authority(scenario, 1, 0)
