"""Guarded drain recognizes baseline work already admitted in a later phase."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import baseline_infrastructure_continuation485_postgres_test as prior
from tests.lab_arena import owner_stopped_execute_drain461_postgres_test as stopped

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/486-lab-arena-restart-baseline-continuation-phase.sql'
base = prior.base
guard = stopped.guard


def active(conn, phase='stage2_scoring'):
    prior.seed(conn, phase)
    assert base.recover(conn)['queued'] == 1
    prior.cool(conn)
    lease = prior.priority.claim(conn)
    assert (lease['stage'], lease['kind'], lease['attempt']) == (1, 'execute', 4)
    return lease


def snapshot(conn):
    with conn.cursor() as cur:
        result = []
        for table, order in [('lab_arena_rounds', 'round_id'), ('lab_arena_runs', 'run_id'),
                             ('lab_arena_ledger', 'entry_id'),
                             ('lab_arena_trajectory_events', 'trajectory_id')]:
            cur.execute(f'SELECT to_jsonb(x) FROM public.{table} x ORDER BY {order}')
            result.append(cur.fetchall())
        return result


def due(conn, lease):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (lease['run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()


def close_deadline(conn):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((clock_timestamp()-interval '1 second')::text))")
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()


def capture(conn):
    return stopped.prior._hold_capture(conn)


@pytest.fixture(scope='module')
def database():
    generator = prior.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as conn:
            conn.autocommit = True
            lease = active(conn)
            generation = capture(conn)
            original = snapshot(conn)
            with pytest.raises(db[0].Error, match='stopped score drain captured write set differs') as error:
                guard._quiescence(conn, generation=generation)
            assert error.value.pgcode == '55000'
            conn.rollback()
            assert snapshot(conn) == original
            with conn.cursor() as cur:
                before = prior.functions(cur)
                text = SQL.read_text()
                for name in ('v_preimages', 'v_postimages'):
                    value = text.split(name + ' TEXT[] := ARRAY[\'')[1].split("'")[0]
                    with pytest.raises(db[0].Error, match='preimage differs' if name == 'v_preimages' else 'postimage differs'):
                        cur.execute(text.replace(value, '0' * 64, 1))
                    conn.rollback()
                    assert prior.functions(cur) == before
                cur.execute('BEGIN')
                cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_expire_leases(text) TO authenticated')
                with pytest.raises(db[0].Error, match='security shape differs'):
                    cur.execute(text)
                conn.rollback()
                assert prior.functions(cur) == before
                cur.execute(text)
                after = prior.functions(cur)
                assert sum(a[1] != b[1] for a, b in zip(before, after)) == 3
                assert [(r[0], r[2:]) for r in before] == [(r[0], r[2:]) for r in after]
                cur.execute(text)
                assert prior.functions(cur) == after
                cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
                assert cur.fetchone() == (False,)
            waiting = guard._quiescence(conn, generation=generation)
            assert (waiting['still_leased_count'], waiting['expired_receipt_count'], waiting['lost_or_mutated_count']) == (1, 0, 0)
            assert snapshot(conn) == original
        yield db
    finally:
        generator.close()


@pytest.mark.parametrize('phase', ['stage2', 'stage2_closed', 'stage2_scoring'])
@pytest.mark.parametrize('after_close', [False, True])
def test_live_baseline_waits_unchanged_then_completes_normally(database, phase, after_close):
    with database[0].connect(**database[1]) as conn:
        lease = active(conn, phase)
        if after_close:
            close_deadline(conn)
        generation = capture(conn)
        original = snapshot(conn)
        waiting = guard._quiescence(conn, generation=generation)
        assert (waiting['still_leased_count'], waiting['expired_receipt_count'], waiting['preserved']) == (1, 0, False)
        assert snapshot(conn) == original
        assert prior.complete(conn, lease)['status'] == 'accepted'
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['accepted_receipt_count'], drained['expired_receipt_count'], drained['lost_or_mutated_count'], drained['preserved']) == (1, 0, 0, True)
        assert prior.priority.claim(conn)['status'] == 'paused'


@pytest.mark.parametrize('after_close', [False, True])
def test_natural_expiry_retains_exact_captured_authority_after_close(database, after_close):
    with database[0].connect(**database[1]) as conn:
        lease = active(conn)
        generation = capture(conn)
        if after_close:
            close_deadline(conn)
        due(conn, lease)
        before = snapshot(conn)
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['expired_receipt_count'], drained['lost_or_mutated_count'], drained['preserved']) == (1, 0, True)
        after = snapshot(conn)
        assert after[0] == before[0]  # Frozen configuration and round phase.
        assert after[2:] == before[2:]  # No ledger or trajectory rewrite.
        assert guard._quiescence(conn, generation=generation)['outcome_commitment'] == drained['outcome_commitment']
        assert prior.priority.claim(conn)['status'] == 'paused'


@pytest.mark.parametrize('paid', [False, True])
def test_authenticated_stopped_baseline_recovers_and_keeps_receipt_proof(database, paid):
    with database[0].connect(**database[1]) as conn:
        lease = active(conn)
        stopped.paid._event(conn, lease, 'runtime.started', {
            'status': 'starting', 'runtime': 'runsc', 'lease_generation': lease['lease_generation'],
        }, token_hash=base.TOKEN)
        if paid:
            call = 'sha256:' + 'f' * 64
            assert base.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)", (lease['run_id'], base.TOKEN, call))['status'] == 'reserved'
            assert base.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)', (lease['run_id'], base.TOKEN, call))['status'] == 'dispatched'
            assert base.rpc(conn, 'lab_arena_settle_call(%s,%s,%s,100,%s::jsonb,4500)', (lease['run_id'], base.TOKEN, call, json.dumps({'status': 200, 'call_succeeded': True})))['status'] == 'settled'
        stopped.paid._event(conn, lease, 'runtime.error', {
            'status': 'abandoned', 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError',
            'runtime_host_reason': 'sandbox_launcher_signaled', 'launch_exit_code': -9, 'launch_timed_out': False,
        }, token_hash=base.TOKEN)
        generation = capture(conn)
        before = snapshot(conn)
        drained = guard._quiescence(conn, generation=generation)
        # Captured zero-call work retains its existing natural-expiry rule;
        # only fully settled stopped work has owner-authorized early recovery.
        expected = (1, 0, True) if paid else (0, 0, False)
        assert (drained['expired_receipt_count'], drained['lost_or_mutated_count'], drained['preserved']) == expected
        after = snapshot(conn)
        assert after[0] == before[0]
        assert after[2:] == before[2:]


@pytest.mark.parametrize('fault', ['owner', 'guard_generation', 'capture_generation', 'stage_generation',
                                  'miner_phase', 'uncaptured', 'foreign_lease', 'result', 'output'])
@pytest.mark.parametrize('overdue', [False, True])
def test_owner_capture_generation_and_nonbaseline_fences_remain_closed(database, fault, overdue):
    with database[0].connect(**database[1]) as conn:
        lease = active(conn)
        generation = capture(conn)
        if overdue:
            due(conn, lease)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            if fault == 'capture_generation':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases=jsonb_set(captured_leases,'{0,lease_generation}','999')")
            elif fault == 'uncaptured':
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases='[]'")
            elif fault == 'stage_generation':
                cur.execute('UPDATE public.lab_arena_runs SET stage_generation=stage_generation+1 WHERE run_id=%s', (lease['run_id'],))
            elif fault == 'miner_phase':
                cur.execute('UPDATE public.lab_arena_submissions SET is_king=false WHERE submission_id=%s', (base.SUB,))
            elif fault in ('result', 'output'):
                cur.execute('UPDATE public.lab_arena_runs SET ' + ("result_doc='{}'" if fault == 'result' else "output_ref='retained-output'") + ' WHERE run_id=%s', (lease['run_id'],))
            elif fault == 'foreign_lease':
                cur.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(r)||jsonb_build_object('run_id','foreign-run','assignment_id','foreign-assignment','claim_request_id',NULL,'claim_request_hash',NULL))).* FROM public.lab_arena_runs r WHERE run_id=%s", (lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        original = snapshot(conn)
        if fault in ('capture_generation', 'uncaptured'):
            result = guard._quiescence(conn, generation=generation)
            assert result['preserved'] is False
            assert result['expired_receipt_count'] == 0
        else:
            with pytest.raises(database[0].Error, match='owner_or_generation_differs' if fault in ('owner', 'guard_generation') else 'captured write set differs'):
                guard._quiescence(conn, generation=generation + int(fault == 'guard_generation'),
                                  owner='lab_arena_restart_owner:' + 'f' * 64 if fault == 'owner' else guard.OWNER)
            conn.rollback()
        assert snapshot(conn) == original


def test_ordinary_live_execute_phase_still_waits(database):
    with database[0].connect(**database[1]) as conn:
        stopped._reset(conn)
        lease = stopped._seed(conn, abandoned=False)
        generation = capture(conn)
        before = snapshot(conn)
        result = guard._quiescence(conn, generation=generation)
        assert (result['still_leased_count'], result['expired_receipt_count'], result['preserved']) == (1, 0, False)
        assert snapshot(conn) == before
        assert base.rpc(conn, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', (
            lease['run_id'], stopped.paid.TOKEN_HASH,
            json.dumps({'terminal_status': 'accepted'}), 'accepted', 'ordinary-output',
        ))['status'] == 'accepted'
        drained = guard._quiescence(conn, generation=generation)
        assert (drained['accepted_receipt_count'], drained['expired_receipt_count'], drained['preserved']) == (1, 0, True)


@pytest.mark.parametrize('head', ['reservation', 'dispatch', 'uncertain'])
def test_unfinished_provider_accounting_cannot_force_early_drain(database, head):
    with database[0].connect(**database[1]) as conn:
        lease = active(conn)
        call = 'sha256:' + 'f' * 64
        assert base.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','host',100,'{}'::jsonb,4500)", (lease['run_id'], base.TOKEN, call))['status'] == 'reserved'
        if head != 'reservation':
            assert base.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)', (lease['run_id'], base.TOKEN, call))['status'] == 'dispatched'
        if head == 'uncertain':
            assert base.rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)', (
                lease['run_id'], base.TOKEN, call,
                json.dumps({'call_succeeded': True, 'provider_status': 200}),
            ))['status'] == 'uncertain'
        stopped.paid._event(conn, lease, 'runtime.error', {
            'status': 'abandoned', 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError',
            'runtime_host_reason': 'sandbox_launcher_signaled', 'launch_exit_code': -9, 'launch_timed_out': False,
        }, token_hash=base.TOKEN)
        generation = capture(conn)
        before = snapshot(conn)
        waiting = guard._quiescence(conn, generation=generation)
        assert (waiting['still_leased_count'], waiting['expired_receipt_count'], waiting['preserved']) == (1, 0, False)
        assert snapshot(conn) == before
