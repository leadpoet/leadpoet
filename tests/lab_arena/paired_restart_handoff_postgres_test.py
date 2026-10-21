"""Real claim RPCs stay paused across both existing restart phase transitions."""
from concurrent.futures import ThreadPoolExecutor
import pytest
from tests.lab_arena import ordered_claim_admission520_postgres_test as queue
from tests.lab_arena import test_lab_arena_restart_claim_drain_postgres as guard


@pytest.fixture(scope='module')
def database():
    generator = queue.database.__wrapped__()
    try:
        yield next(generator)
    finally:
        generator.close()


@pytest.mark.parametrize('operator_hold', [False, True])
def test_single_guard_blocks_real_claims_until_both_ready_and_preserves_operator_hold(database, operator_hold):
    rid, ids = queue.prepared(database)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE status='accepted'")
        accepted = cur.fetchone()[0]
        cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton')
        prior_generation = cur.fetchone()[0]
        cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=%s WHERE singleton', (operator_hold,))
        acquired = guard._acquire(conn, generation=prior_generation, scope='all')
        generation = acquired['guard_generation']
        assert generation == prior_generation + 1
    def claim():
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            return queue.claim(cur, rid)
    phases = [('lab_arena_authorize_restart_phase_v1', 'gateway_destructive'),
              ('lab_arena_mark_restart_ready_v1', 'gateway_ready'),
              ('lab_arena_authorize_restart_phase_v1', 'validator_destructive'),
              ('lab_arena_mark_restart_ready_v1', 'validator_ready')]
    for function, phase in phases:
        with database[0].connect(**database[1]) as conn:
            renewed = guard._acquire(conn, generation=generation, scope='all')
            assert renewed['guard_generation'] == generation
            result = guard._rpc(conn, function, guard.GUARD, guard.OWNER, generation, phase)
            assert result['restart_phase'] == phase
        with ThreadPoolExecutor(max_workers=3) as pool:
            outcomes = list(pool.map(lambda _: claim(), range(3)))
        assert all(outcome['status'] == 'paused' for outcome in outcomes), outcomes
        if phase != 'validator_ready':
            with database[0].connect(**database[1]) as conn:
                with pytest.raises(database[0].Error):
                    guard._rpc(conn, 'lab_arena_release_restart_guard_v1', guard.GUARD, guard.OWNER, generation, 'early-release')
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE status='accepted'")
            assert cur.fetchone()[0] == accepted
            cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND status='leased'", (rid,))
            assert cur.fetchone() == (0,)
    with database[0].connect(**database[1]) as conn:
        # Neither wrong owner nor stale generation can release the pair.
        for owner, gen in [(guard.OWNER+'x', generation), (guard.OWNER, generation+1)]:
            with pytest.raises(database[0].Error):
                guard._rpc(conn, 'lab_arena_release_restart_guard_v1', guard.GUARD, owner, gen, 'wrong-owner')
            conn.rollback()
        released = guard._rpc(conn, 'lab_arena_release_restart_guard_v1', guard.GUARD, guard.OWNER, generation, 'pair-ready')
        assert released['guard_present'] is False
        assert released['operator_paused'] is operator_hold
    result = claim()
    assert result['status'] == ('paused' if operator_hold else 'leased')
    if not operator_hold:
        assert result['run_id'] == ids[0]
