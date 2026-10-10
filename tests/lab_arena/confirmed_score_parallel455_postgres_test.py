"""Real PostgreSQL proof for current-policy parallel score claims."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import stage_cutoff_drain420_postgres_test as prior
from tests.lab_arena.confirmed_cost_admission_postgres_test import _open_current_scoring
from tests.lab_arena.score_submission_serialization_postgres_test import _open_scoring, _reserve_dynamic
from tests.lab_arena.test_lab_arena_migration_postgres import claim

base_database = prior.base_database
current_database = prior.current_database
database = prior.database
SQL = Path(__file__).parents[2] / 'scripts/455-lab-arena-confirmed-score-parallel.sql'
SIG = 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
PRE = '990baa411ac35d075ded6a56f14c5bd23c72416f85706eab4a802e3f08f51b3e'
POST = 'cd889cc33098309a32effa193f6eb389b26d6ac1beaa0f448b6a01b7edf2b16e'


def _definition(cur):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
    return cur.fetchone()[0]


def _security(cur):
    cur.execute('SELECT proowner,proacl,prosecdef,provolatile,proconfig FROM pg_proc WHERE oid=%s::regprocedure', (SIG,))
    return cur.fetchone()


@pytest.fixture(scope='module')
def migrated(database):
    pg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    store, runners, _ = _open_current_scoring(store, 'arena-2099-03-21')
    active, token = claim(store, 'arena-2099-03-21', runners[0], parallelism=8, ceiling=8, excluded=[runners[0]])[:2]
    assert active['status'] == 'leased'
    before = store.get_run(active['run_id'])
    with pg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            original = _definition(cur)
            security = _security(cur)
            cur.execute('BEGIN')
            with pytest.raises(pg.Error, match='preimage differs'):
                cur.execute(SQL.read_text().replace(PRE, '0'*64))
            cur.execute('ROLLBACK')
            assert _definition(cur) == original
            cur.execute('BEGIN')
            with pytest.raises(pg.Error, match='postimage differs'):
                cur.execute(SQL.read_text().replace(POST, '0'*64))
            cur.execute('ROLLBACK')
            assert _definition(cur) == original
            cur.execute(SQL.read_text())
            after = _definition(cur)
            cur.execute(SQL.read_text())
            assert _definition(cur) == after
            assert _security(cur) == security
            # Exact textual delta: every authority, worker, budget and cutoff
            # guard outside the old serialization predicate is unchanged.
            addition = "      -- lab_arena_confirmed_score_parallel_v1: current-policy admission and\n      -- settlement share the submission lock; pending calls hold no money.\n      OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'\n           = 'successful_calls_per_icp_v1'\n"
            assert after.replace(addition, '') == original
    assert store.get_run(active['run_id']) == before
    yield store, runners, active, token
    store.close()


def test_existing_live_lease_is_preserved_and_sibling_can_claim(database, migrated):
    store, runners, active, _ = migrated
    sibling = claim(store, 'arena-2099-03-21', runners[1], parallelism=8, ceiling=8, excluded=[runners[1]])[0]
    assert sibling['status'] == 'leased'
    assert sibling['submission_id'] == active['submission_id']
    assert sibling['run_id'] != active['run_id']


def _race(database, round_id, runners):
    barrier = threading.Barrier(2)
    def worker(index):
        pg, dsn = database
        store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
        try:
            barrier.wait(timeout=10)
            return claim(store, round_id, runners[index], parallelism=8, ceiling=8, excluded=[runners[index]])[:2]
        finally:
            store.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(worker, (0, 1)))


def test_concurrent_siblings_use_atomic_confirmed_cap(database, migrated):
    pg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    try:
        _, runners, _ = _open_current_scoring(store, 'arena-2099-03-22')
        raced = _race(database, 'arena-2099-03-22', runners)
        assert all(row['status'] == 'leased' for row, _ in raced)
        assert len({row['run_id'] for row, _ in raced}) == 2
        assert len({row['submission_id'] for row, _ in raced}) == 1
        first, first_token = raced[0]
        second, second_token = raced[1]
        identity, reserved = _reserve_dynamic(store, first, first_token, 'parallel-first')
        assert (reserved['status'], reserved['amount_microusd']) == ('reserved', 0)
        other, admitted = _reserve_dynamic(store, second, second_token, 'parallel-second')
        assert (admitted['status'], admitted['amount_microusd']) == ('reserved', 0)
        assert store.mark_dispatched(run_id=first['run_id'], lease_token_hash=hash_lease_token(first_token), call_identity=identity)['status'] == 'dispatched'
        assert store.settle_call(run_id=first['run_id'], lease_token_hash=hash_lease_token(first_token), call_identity=identity, actual_microusd=50_000_000, terminal_response={'status':200,'call_succeeded':True})['status'] == 'settled'
        _, blocked = _reserve_dynamic(store, second, second_token, 'parallel-after-cap')
        assert (blocked['status'], blocked['reason']) == ('refused', 'money_cap')
    finally:
        store.close()


def test_concurrent_same_job_has_one_lease(database, migrated):
    pg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    try:
        _, runners, _ = _open_current_scoring(store, 'arena-2099-03-23')
        # A disposable fixture leaves exactly one eligible job; no production
        # state is changed and the actual claim function handles the race.
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("DELETE FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND icp_position<>0", ('arena-2099-03-23',))
        raced = _race(database, 'arena-2099-03-23', runners)
        assert sorted(row['status'] for row, _ in raced) == ['leased', 'no_pending']
    finally:
        store.close()


def test_legacy_siblings_remain_serial(database, migrated):
    pg, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    try:
        runners, _ = _open_scoring(store, 'arena-2099-03-24', participants=1, runners=2)
        raced = _race(database, 'arena-2099-03-24', runners)
        assert sorted(row['status'] for row, _ in raced) == ['leased', 'no_pending']
    finally:
        store.close()


@pytest.mark.parametrize('mutation,error', [
    ('acl', 'security shape differs'),
    ('prerequisite', 'admission prerequisite differs'),
    ('body', 'preimage differs'),
])
def test_replay_rejects_changed_security_prerequisite_or_body(database, migrated, mutation, error):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            original = _definition(cur)
            security = _security(cur)
            cur.execute('BEGIN')
            if mutation == 'acl':
                cur.execute(f'GRANT EXECUTE ON FUNCTION {SIG} TO PUBLIC')
            elif mutation == 'prerequisite':
                cur.execute("SELECT pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure)")
                reserve = cur.fetchone()[0]
                cur.execute(reserve.replace('lab_arena_confirmed_score_admission', 'missing_score_admission'))
            else:
                cur.execute(original.replace('lab_arena_confirmed_score_parallel_v1', 'unexpected_score_parallel_v1'))
            with pytest.raises(pg.Error, match=error):
                cur.execute(SQL.read_text())
            cur.execute('ROLLBACK')
            assert _definition(cur) == original
            assert _security(cur) == security
