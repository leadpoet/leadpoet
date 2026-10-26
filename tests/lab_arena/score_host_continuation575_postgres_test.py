"""Stopped score leases recover across real stage advancement, once only."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena.store import hash_lease_token
from tests.lab_arena import concurrent_baseline_judging572_postgres_test as prior
from tests.lab_arena import company_judgment_validation_projection548_postgres_test as completed

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/575-lab-arena-score-host-continuation.sql'
SIG = 'public.lab_arena_expire_leases(text)'
PRE = 'cbe667688a31c3e0fbc39e968721b92ff206780c6818df8afde4eeaedd98c187'
POST = 'f8b2d491b97404f6dbea55cedeac2fb277897b9bdd7d67447b4fd43e56340a83'


@pytest.fixture(scope='module')
def scenario(tmp_path_factory):
    from scripts.verify_arena_full_thinking_e2e import full_thinking_database
    database = full_thinking_database()
    db = next(database)
    fixture = completed.completed.__wrapped__(db, tmp_path_factory)
    try:
        h, row = next(fixture)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            assert digest(cur) == PRE
            saved = {}
            for table in ('lab_arena_rounds', 'lab_arena_runs', 'lab_arena_ledger',
                          'lab_arena_judgment_cache'):
                cur.execute(f'SELECT to_jsonb(t) FROM public.{table} t')
                saved[table] = [item[0] for item in cur.fetchall()]
        yield db, h, row, saved
    finally:
        fixture.close()
        database.close()


def digest(cur):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
    return hashlib.sha256(cur.fetchone()[0].encode()).hexdigest()


@pytest.fixture(autouse=True)
def restore(scenario):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
        source = cur.fetchone()[0]
        cur.execute('TRUNCATE public.lab_arena_trajectory_events CASCADE')
        cur.execute('TRUNCATE public.lab_arena_company_judgment_reservations CASCADE')
    yield
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute(source)


def prepare(scenario, *, error=True):
    db, h, _, _ = scenario
    rid, _, pending = prior.prepare(scenario)
    # The saved accepted fixture reuses attempt 1's identifier. Remove its
    # historical test bills before making this genuinely new zero-call claim.
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('DELETE FROM public.lab_arena_ledger WHERE run_id=%s', (pending['run_id'],))
    runner = h.runner(0)
    try:
        lease = runner.claim_one()
        assert lease['kind'] == 'score' and lease['stage'] == 1
        assert h.service.advance_round(rid)['status'] == 'ok'
        assert h.status() == 'stage2'
        if error:
            now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            batch = [{'event_id': str(uuid.uuid4()), 'kind': 'runtime.error',
                      'occurred_at': now, 'content': {
                          'status': 'abandoned', 'failure_stage': 'runtime',
                          'error_class': 'RuntimeHostError',
                          'runtime_host_reason': 'sandbox_launch_failed',
                          'launch_exit_code': 128, 'launch_timed_out': False}}]
            with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
                cur.execute('SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)',
                            (lease['run_id'], hash_lease_token(lease['lease_token']), json.dumps(batch)))
                assert cur.fetchone()[0]['status'] == 'accepted'
        return lease
    finally:
        runner.close()


def expire(cur, lease):
    return prior.scalar(cur, 'public.lab_arena_expire_leases(%s)', (lease['round_id'],))


def state(cur, lease):
    return prior.scalar(cur, 'to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s',
                        (lease['run_id'],))


def test_real_stage_transition_fails_before_and_recovers_once_after(scenario):
    db, h, _, _ = scenario
    lease = prepare(scenario)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        before = state(cur, lease)
        assert prior.scalar(cur, 'count(*) FROM public.lab_arena_ledger WHERE run_id=%s',
                            (lease['run_id'],)) == 0
        assert prior.scalar(cur, 'public.lab_arena__judge_continuation_v1(d,r) '
                            'FROM public.lab_arena_rounds d JOIN public.lab_arena_runs r USING(round_id) '
                            'WHERE r.run_id=%s', (lease['run_id'],)) is True
        assert before['stage_generation'] < h.service.store.get_round(lease['round_id'])['stage_generation']
        assert expire(cur, lease) == {'status': 'ok', 'expired': 0, 'retried': 0}
        frozen = prior.prior.fingerprint(cur, lease['round_id'])
        security = prior.scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure", (SIG,))
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        assert digest(cur) == POST
        assert state(cur, lease) == before
        assert prior.scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure", (SIG,)) == security
        assert expire(cur, lease) == {'status': 'ok', 'expired': 1, 'retried': 1}
        assert expire(cur, lease) == {'status': 'ok', 'expired': 0, 'retried': 0}
        after = state(cur, lease)
        assert after['status'] == 'failed' and after['terminal_cause'] == 'lease_expired'
        assert after['claim_response'] == before['claim_response']
        assert after['terminal_doc']['recovery_reason'] == 'authenticated_zero_call_runtime_host_error'
        assert prior.prior.fingerprint(cur, lease['round_id']) == frozen
        cur.execute('SELECT status,scored_run_id,stage_generation FROM public.lab_arena_runs '
                    'WHERE assignment_id=%s AND attempt=%s', (lease['assignment_id'], lease['attempt'] + 1))
        assert cur.fetchone() == ('pending', before['scored_run_id'], before['stage_generation'] + 1)


@pytest.mark.parametrize('control', ['active', 'paid', 'wrong_receipt', 'wrong_floor', 'execute'])
def test_recovery_keeps_existing_fences(scenario, control):
    db = scenario[0]
    lease = prepare(scenario, error=control != 'active')
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        if control == 'paid':
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) "
                        "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','host',100 FROM public.lab_arena_runs WHERE run_id=%s",
                        ('sha256:' + 'f' * 64, lease['run_id']))
        elif control == 'wrong_receipt':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET attempt=attempt+1 WHERE run_id=%s", (lease['run_id'],))
        elif control == 'wrong_floor':
            cur.execute('UPDATE public.lab_arena_runs SET scoring_generation_floor=scoring_generation_floor+1 WHERE run_id=%s', (lease['run_id'],))
        elif control == 'execute':
            cur.execute("UPDATE public.lab_arena_runs SET kind='execute',scored_run_id=NULL,"
                        "company_judgment_refs=NULL,claim_response=jsonb_set(claim_response,'{kind}',"
                        "'\"execute\"'::jsonb) WHERE run_id=%s", (lease['run_id'],))
            cur.execute("UPDATE public.lab_arena_trajectory_events SET run_kind='execute' WHERE run_id=%s", (lease['run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute(SQL.read_text())
        before = state(cur, lease)
        assert expire(cur, lease)['expired'] == 0
        assert state(cur, lease) == before


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source', 'acl', 'security'])
def test_migration_rejects_unknown_function_or_security(scenario, installed, drift):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        if installed:
            cur.execute(SQL.read_text())
        cur.execute('BEGIN')
        cur.execute('SAVEPOINT drift575')
        if drift == 'source':
            cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIG,))
            cur.execute(cur.fetchone()[0].replace('BEGIN', 'BEGIN\n  NULL;', 1))
        elif drift == 'acl':
            cur.execute('GRANT EXECUTE ON FUNCTION ' + SIG + ' TO anon')
        else:
            cur.execute('ALTER FUNCTION ' + SIG + ' SECURITY INVOKER')
        body = SQL.read_text().split('DO $score_host_continuation575$')[1].split('$score_host_continuation575$;')[0]
        with pytest.raises(db[0].Error):
            cur.execute('DO $test$' + body + '$test$')
        cur.execute('ROLLBACK TO SAVEPOINT drift575')
