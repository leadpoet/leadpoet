"""Real provider -> sweep -> claim contention, with recovery fences intact."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import pytest

from tests.lab_arena import partial_judge_host_fault527_postgres_test as prior
from tests.lab_arena import progression_lock_order490_postgres_test as locks
from tests.lab_arena import settled_score_host_recovery441_postgres_test as paid

SQL = Path(__file__).parents[2] / 'scripts/551-lab-arena-expire-leases-shared-round-lock.sql'
SIG = 'public.lab_arena_expire_leases(text)'
PRE = '61e1dd5c5c472ce8060d356794bd9f621a8fc0f579ce091941826e19bf261468'
POST = 'd1ffdf531e598254d3fd4241e44ca2a1d5721580e559d1c613ad8ccfd3b3f0ad'
ROUND = prior.started.fast.seed.ROUND


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        yield db
    finally:
        gen.close()


def definition(q):
    q.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
    return q.fetchone()[0]


def apply(c):
    with c.cursor() as q:
        q.execute(SQL.read_text())
    c.commit()


@pytest.fixture(autouse=True)
def restore(database):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        saved = definition(q)
        assert hashlib.sha256(saved.encode()).hexdigest() == PRE
    yield
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        q.execute(saved)
        q.execute('TRUNCATE public.lab_arena_rounds CASCADE')
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,"
                  "pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',"
                  "guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]'")


def catalog(q):
    q.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' "
              "FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
    return {oid: (source, metadata) for oid, source, metadata in q.fetchall()}


def test_exact_migration_replay_and_metadata_rows_unchanged(database):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        prior.cohort(c, count=1, extra=1, started=1)
        before = catalog(q)
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r')
        rows = q.fetchone()[0]
        q.execute('CREATE ROLE lock551_migrator NOLOGIN NOSUPERUSER INHERIT')
        q.execute('GRANT lab_arena_owner TO lock551_migrator')
        q.execute('GRANT USAGE,CREATE ON SCHEMA public TO lock551_migrator')
        q.execute('SET ROLE lock551_migrator')
        q.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert q.fetchone() == (False,)
        apply(c)
        apply(c)
        q.execute('RESET ROLE')
        after = catalog(q)
        changed = [oid for oid in before if before[oid] != after[oid]]
        assert len(changed) == 1
        old, metadata = before[changed[0]]
        new, new_metadata = after[changed[0]]
        assert metadata == new_metadata
        assert hashlib.sha256(old.encode()).hexdigest() == PRE
        assert hashlib.sha256(new.encode()).hexdigest() == POST
        assert new == old.replace('WHERE round_id = p_round_id FOR UPDATE;',
                                  'WHERE round_id = p_round_id FOR SHARE;', 1)
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r')
        assert q.fetchone()[0] == rows


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'path', 'security', 'volatility'])
def test_unknown_source_and_security_drift_rejected(database, installed, drift):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        if installed:
            apply(c)
        q.execute('SAVEPOINT hostile')
        if drift == 'source':
            q.execute(definition(q).replace('BEGIN\n', 'BEGIN\n  NULL;\n', 1))
        else:
            q.execute({
                'owner': 'ALTER FUNCTION '+SIG+' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon',
                'path': 'ALTER FUNCTION '+SIG+' SET search_path=public,pg_catalog',
                'security': 'ALTER FUNCTION '+SIG+' SECURITY INVOKER',
                'volatility': 'ALTER FUNCTION '+SIG+' STABLE',
            }[drift])
        body = SQL.read_text().split('DO $expire_leases_shared_round_lock551$')[1].split('$expire_leases_shared_round_lock551$;')[0]
        with pytest.raises(database[0].Error):
            q.execute('DO $test$'+body+'$test$')
        q.execute('ROLLBACK TO SAVEPOINT hostile')


def reserve(c, lease, *, commit=True):
    return paid._rpc(c,
        "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','miner_key',100,'{}'::jsonb,4500)",
        (lease['run_id'], paid.TOKEN_HASH, paid.CALL), commit=commit)


def async_rpc(pool, pg, dsn, expression, args):
    c = pg.connect(**dsn)
    with c.cursor() as q:
        q.execute("SET statement_timeout='5s'")
    def run():
        try:
            return paid._rpc(c, expression, args)
        finally:
            c.close()
    return c.get_backend_pid(), pool.submit(run)


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('provider_phase', ['reserve', 'dispatch', 'complete'])
def test_real_provider_sweep_claim_queue_before_and_after(database, installed, provider_phase):
    pg, dsn = database
    with pg.connect(**dsn) as setup:
        lease = prior.cohort(setup, count=1, extra=1, started=1)[0]
        if installed:
            apply(setup)
        if provider_phase == 'dispatch':
            assert reserve(setup, lease)['status'] == 'reserved'
    with pg.connect(**dsn) as provider, ThreadPoolExecutor(max_workers=2) as pool:
        if provider_phase == 'reserve':
            assert reserve(provider, lease, commit=False)['status'] == 'reserved'
        elif provider_phase == 'dispatch':
            assert paid._dispatch(provider, lease, commit=False)['status'] == 'dispatched'
        else:
            result = paid._rpc(provider,
                'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,NULL)',
                (lease['run_id'], paid.TOKEN_HASH, '{"terminal_status":"judge_error"}',
                 'judge_error'), commit=False)
            assert result['status'] == 'failed'
        sweep_pid, sweep = async_rpc(pool, pg, dsn, 'lab_arena_expire_leases(%s)', (ROUND,))
        if not installed:
            locks._wait(pg, dsn, sweep_pid, None)
        claim_pid, claim = async_rpc(pool, pg, dsn,
            'lab_arena_claim_assignment(%s,%s,7,20,%s,%s,%s,%s,4500)',
            (ROUND, prior.started.fast.seed.RUNNER, [], 'b'*32, 'sha256:'+'b'*64, 'sha256:'+'b'*64))
        if installed:
            assert sweep.result(timeout=2) == {'status': 'ok', 'expired': 0, 'retried': 0}
            assert claim.result(timeout=2)['status'] == 'leased'
            # The actual provider transaction remains open throughout both RPCs.
            assert provider.status == pg.extensions.STATUS_IN_TRANSACTION
        else:
            locks._wait(pg, dsn, claim_pid, 'advisory')
            assert not sweep.done() and not claim.done()
        provider.commit()
        assert sweep.result(timeout=3) == {'status': 'ok', 'expired': 0, 'retried': 0}
        assert claim.result(timeout=3)['status'] == 'leased'


def test_two_real_sweeps_expire_once_and_keep_terminal_and_provider_history(database):
    pg, dsn = database
    with pg.connect(**dsn) as c, c.cursor() as q:
        lease = prior.cohort(c, count=1, extra=0, started=1)[0]
        assert reserve(c, lease)['status'] == 'reserved'
        assert paid._dispatch(c, lease)['status'] == 'dispatched'
        assert paid._settle(c, lease)['status'] == 'settled'
        apply(c)
        prior.started.expire_clock(c)
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY entry_id) FROM public.lab_arena_ledger r')
        ledger = q.fetchone()[0]
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY event_id) FROM public.lab_arena_trajectory_events r')
        events = q.fetchone()[0]
        c.commit()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [async_rpc(pool, pg, dsn, 'lab_arena_expire_leases(%s)', (ROUND,))[1] for _ in range(2)]
        results = [f.result(timeout=4) for f in results]
    assert sorted(r['expired'] for r in results) == [0, 1]
    assert sorted(r['retried'] for r in results) == [0, 1]
    with pg.connect(**dsn) as c, c.cursor() as q:
        q.execute('SELECT status,terminal_cause FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
        assert q.fetchone() == ('failed', 'lease_expired')
        q.execute("SELECT attempt,status FROM public.lab_arena_runs WHERE assignment_id=%s ORDER BY attempt", (lease['assignment_id'],))
        assert q.fetchall() == [(1, 'failed'), (2, 'pending')]
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY entry_id) FROM public.lab_arena_ledger r')
        assert q.fetchone()[0] == ledger
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY event_id) FROM public.lab_arena_trajectory_events r')
        assert q.fetchone()[0] == events
        # A late real completion cannot replace the expiry or add another retry.
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r')
        rows = q.fetchone()[0]
        result = paid._rpc(c,
            'lab_arena_complete_attempt_v3(%s,%s,%s::jsonb,%s,NULL,NULL,%s::jsonb,%s)',
            (lease['run_id'], paid.TOKEN_HASH, '{"terminal_status":"judge_error"}',
             'judge_error', '[]', 'sha256:'+'c'*64))
        assert result['status'] == 'failed' and result['idempotent'] is True
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r')
        assert q.fetchone()[0] == rows


def test_shared_sweep_still_excludes_round_update_and_real_stage_close(database):
    pg, dsn = database
    with pg.connect(**dsn) as c:
        prior.cohort(c, count=1, extra=0, started=1)
        apply(c)
    with pg.connect(**dsn) as sweep, pg.connect(**dsn) as progression, ThreadPoolExecutor(max_workers=1) as pool:
        assert paid._rpc(sweep, 'lab_arena_expire_leases(%s)', (ROUND,), commit=False) == {'status': 'ok', 'expired': 0, 'retried': 0}
        with progression.cursor() as q:
            with pytest.raises(pg.Error) as error:
                q.execute('SELECT 1 FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE NOWAIT', (ROUND,))
            assert error.value.pgcode == '55P03'
        progression.rollback()
        pid, closer = async_rpc(pool, pg, dsn, 'lab_arena_close_stage(%s,2::smallint)', (ROUND,))
        locks._wait(pg, dsn, pid, 'advisory')
        sweep.commit()
        assert closer.result(timeout=3)['status'] == 'existing'


@pytest.mark.parametrize('fault', ['none', 'no_start', 'open_dispatch', 'unbound_error', 'setup_image', 'setup_runtime'])
def test_owner_recovery_keeps_existing_authentication_and_liability_fences(database, fault):
    with database[0].connect(**database[1]) as c:
        apply(c)
    prior.recovery.gate.test_signal_keeps_full_stopped_host_authorization(database, fault)
