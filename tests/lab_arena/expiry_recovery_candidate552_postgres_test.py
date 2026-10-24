"""An unrelated host error does not serialize all healthy provider work."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import pytest

from tests.lab_arena import expire_leases_shared_round551_postgres_test as prior

SQL = Path(__file__).parents[2] / 'scripts/552-lab-arena-expiry-recovery-candidate-locks.sql'
POST = '5025e77ac2a602941e7166d0a201ba49efb03d0da8af1a0fda1e1889056de40d'
database = prior.database
restore = prior.restore


def apply(c):
    prior.apply(c)
    with c.cursor() as q:
        q.execute(SQL.read_text())
    c.commit()


def test_exact_replay_metadata_and_all_other_functions_and_rows_unchanged(database):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        prior.prior.cohort(c, count=1, extra=1, started=1)
        prior.apply(c)
        before = prior.catalog(q)
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r')
        rows = q.fetchone()[0]
        q.execute('CREATE ROLE candidate552_migrator NOLOGIN NOSUPERUSER INHERIT')
        q.execute('GRANT lab_arena_owner TO candidate552_migrator')
        q.execute('GRANT USAGE,CREATE ON SCHEMA public TO candidate552_migrator')
        q.execute('SET ROLE candidate552_migrator')
        q.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert q.fetchone() == (False,)
        q.execute(SQL.read_text())
        q.execute(SQL.read_text())
        q.execute('RESET ROLE')
        after = prior.catalog(q)
        changed = [oid for oid in before if before[oid] != after[oid]]
        assert len(changed) == 1
        old, metadata = before[changed[0]]
        new, new_metadata = after[changed[0]]
        assert new_metadata == metadata
        assert hashlib.sha256(old.encode()).hexdigest() == prior.POST
        assert hashlib.sha256(new.encode()).hexdigest() == POST
        assert new.count("candidate_error.event_kind='runtime.error'") == 2
        filter_text = "      AND EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events candidate_error\n        WHERE candidate_error.run_id=candidate.run_id\n          AND candidate_error.event_kind='runtime.error')\n"
        assert new.replace(filter_text, '') == old
        q.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r')
        assert q.fetchone()[0] == rows


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'path', 'security', 'volatility'])
def test_unknown_source_and_security_drift_rejected(database, installed, drift):
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        prior.apply(c)
        if installed:
            q.execute(SQL.read_text())
            c.commit()
        q.execute('SAVEPOINT hostile')
        if drift == 'source':
            q.execute(prior.definition(q).replace('BEGIN\n', 'BEGIN\n  NULL;\n', 1))
        else:
            q.execute({
                'owner': 'ALTER FUNCTION '+prior.SIG+' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION '+prior.SIG+' TO anon',
                'path': 'ALTER FUNCTION '+prior.SIG+' SET search_path=public,pg_catalog',
                'security': 'ALTER FUNCTION '+prior.SIG+' SECURITY INVOKER',
                'volatility': 'ALTER FUNCTION '+prior.SIG+' STABLE',
            }[drift])
        body = SQL.read_text().split('DO $expiry_recovery_candidate_locks552$')[1].split('$expiry_recovery_candidate_locks552$;')[0]
        with pytest.raises(database[0].Error):
            q.execute('DO $test$'+body+'$test$')
        q.execute('ROLLBACK TO SAVEPOINT hostile')


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('error_status', ['judge_error', 'abandoned'])
def test_unrelated_runtime_error_does_not_lock_healthy_provider_or_block_claims(database, installed, error_status):
    pg, dsn = database
    with pg.connect(**dsn) as c:
        leases = prior.prior.cohort(c, count=2, extra=1, started=2)
        prior.prior.started.event(c, leases[0], 'a', kind='runtime.error', content={
            'status': error_status, 'failure_stage': 'runtime', 'error_class': 'RuntimeHostError'})
        prior.apply(c)
        if installed:
            with c.cursor() as q:
                q.execute(SQL.read_text())
            c.commit()
    with pg.connect(**dsn) as provider, ThreadPoolExecutor(max_workers=2) as pool:
        assert prior.paid._rpc(provider,
            "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter','miner_key',100,'{}'::jsonb,4500)",
            (leases[1]['run_id'], 'sha256:'+'b'*64, prior.paid.CALL), commit=False)['status'] == 'reserved'
        sweep_pid, sweep = prior.async_rpc(pool, pg, dsn, 'lab_arena_expire_leases(%s)', (prior.ROUND,))
        if not installed:
            prior.locks._wait(pg, dsn, sweep_pid, None)
        claim_pid, claim = prior.async_rpc(pool, pg, dsn,
            'lab_arena_claim_assignment(%s,%s,7,20,%s,%s,%s,%s,4500)',
            (prior.ROUND, prior.prior.started.fast.seed.RUNNER, [], 'c'*32, 'sha256:'+'c'*64, 'sha256:'+'c'*64))
        if installed:
            expected = int(error_status == 'abandoned')
            assert sweep.result(timeout=2) == {'status': 'ok', 'expired': expected, 'retried': expected}
            assert claim.result(timeout=2)['status'] == 'leased'
        else:
            prior.locks._wait(pg, dsn, claim_pid, 'advisory')
            assert not sweep.done() and not claim.done()
        provider.commit()
        assert sweep.result(timeout=3)['status'] == 'ok'
        assert claim.result(timeout=3)['status'] == 'leased'


@pytest.mark.parametrize('fault', ['none', 'no_start', 'open_dispatch', 'unbound_error', 'setup_image', 'setup_runtime'])
def test_existing_owner_recovery_authentication_and_liability_fences(database, fault):
    with database[0].connect(**database[1]) as c:
        apply(c)
    prior.prior.recovery.gate.test_signal_keeps_full_stopped_host_authorization(database, fault)
