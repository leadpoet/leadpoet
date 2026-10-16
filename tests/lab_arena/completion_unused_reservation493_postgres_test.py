"""Finished attempts recover only call reservations that were never dispatched."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
from unittest.mock import patch

import pytest

from tests.lab_arena import baseline_infrastructure_continuation485_postgres_test as previous
from tests.lab_arena import bounded_baseline_recovery470_postgres_test as base
from tests.lab_arena import general_judge_continuation472_postgres_test as judge
from tests.lab_arena import host_interruption_retry474_postgres_test as execution

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/493-lab-arena-completion-unused-reservation-recovery.sql'
TOKEN = base.TOKEN


def seed(conn, *, kind='execute', baseline=False):
    if kind == 'execute':
        round_id = execution.ROUND
        submission = 'baseline-' + round_id.removeprefix('arena-') if baseline else execution.cutoff.SUB
        with patch.object(execution.cutoff, 'SUB', submission):
            execution.seed(conn)
    else:
        assignment, _, _ = judge.prepare(conn)
        round_id, submission = base.ROUND, base.SUB
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_submissions SET is_king=%s WHERE submission_id=%s',
                    (baseline, submission))
        cur.execute('SELECT miner_hotkey FROM public.lab_arena_submissions WHERE submission_id=%s',
                    (submission,))
        miner = cur.fetchone()[0]
        cur.execute("UPDATE public.lab_arena_rounds SET participants=%s::jsonb,configuration_doc=jsonb_set(configuration_doc,'{baseline_hotkey}',%s::jsonb) WHERE round_id=%s",
                    (json.dumps([dict(submission_id=submission, miner_hotkey=miner, is_king=baseline)]
                      + ([] if baseline else [dict(submission_id='other-baseline', miner_hotkey='other-owner', is_king=True)])),
                     json.dumps(miner if baseline else 'other-owner'), round_id))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    if kind == 'execute':
        lease = execution.claim(conn)
    else:
        assert base.rpc(conn, 'lab_arena_expire_leases(%s)', (round_id,))['retried'] == 1
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE run_id=%s",
                        (assignment + ':3',))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        lease = base.claim(conn)
    assert lease['status'] == 'leased', lease
    return lease


def complete(conn, lease, cause='accepted', *, token=TOKEN, commit=True):
    result = {'terminal_status': cause}
    args = (lease['run_id'], token, json.dumps(result), cause,
            'retained-output' if cause == 'accepted' else '')
    if commit:
        return base.rpc(conn, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', args)
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)', args)
        return cur.fetchone()[0]


def reserve(conn, lease, suffix='a', *, amount=123, funding=None):
    if funding is None:
        with conn.cursor() as cur:
            cur.execute('SELECT s.is_king FROM public.lab_arena_runs r JOIN public.lab_arena_submissions s USING(submission_id) WHERE r.run_id=%s',
                        (lease['run_id'],))
            funding = 'host' if cur.fetchone()[0] else 'miner_key'
    identity = 'sha256:' + hashlib.sha256((lease['run_id'] + suffix).encode()).hexdigest()
    result = base.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter',%s,%s,'{}'::jsonb,4500)",
                      (lease['run_id'], TOKEN, identity, funding, amount))
    assert result['status'] == 'reserved', result
    return identity


def dispatch(conn, lease, identity, *, commit=True):
    args = (lease['run_id'], TOKEN, identity)
    if commit:
        return base.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)', args)
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena_mark_dispatched(%s,%s,%s)', args)
        return cur.fetchone()[0]


def ledger(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id')
        return [row[0] for row in cur.fetchall()]


def score_claim(conn, marker):
    digest = hashlib.sha256(marker.encode()).hexdigest()
    return base.rpc(conn, 'lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,4500)',
                    (base.ROUND, base.RUNNER, digest[:32], 'sha256:' + digest, TOKEN))


def legacy_cost_policy(conn, lease):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc-'sourcing_cost_eligibility_policy' WHERE round_id=(SELECT round_id FROM public.lab_arena_runs WHERE run_id=%s)",
                    (lease['run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()


@pytest.fixture(scope='module')
def database():
    generator = previous.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            # Reproduce the old completion stall and the SQL NULL validation gap.
            lease = seed(conn)
            reserve(conn, lease)
            before = ledger(conn)
            assert complete(conn, lease, 'provider_error') == {'status': 'accounting_open', 'open_calls': 1}
            assert complete(conn, lease, None) == {'status': 'accounting_open', 'open_calls': 1}
            assert ledger(conn) == before
            functions = previous.functions(cur)
            text = SQL.read_text()
            for key, message in [('v_preimage', 'preimage differs'), ('v_postimage', 'postimage differs')]:
                digest = text.split(key + " CONSTANT TEXT := '")[1].split("'")[0]
                with pytest.raises(db[0].Error, match=message):
                    cur.execute(text.replace(digest, '0' * 64, 1))
                conn.rollback()
                assert previous.functions(cur) == functions
            cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_complete_attempt(text,text,jsonb,text,text) TO anon')
            with pytest.raises(db[0].Error, match='security shape differs'):
                cur.execute(text)
            conn.rollback()
            assert previous.functions(cur) == functions
            cur.execute('CREATE ROLE completion493_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO completion493_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO completion493_migrator')
            cur.execute('SET ROLE completion493_migrator')
            cur.execute(text)
            after = previous.functions(cur)
            cur.execute(text)
            assert previous.functions(cur) == after
            cur.execute('RESET ROLE')
            assert [(r[0], r[2:]) for r in functions] == [(r[0], r[2:]) for r in after]
            assert [a[0] for a, b in zip(functions, after) if a[1] != b[1]] == [
                'lab_arena_complete_attempt(text,text,jsonb,text,text)']
        yield db
    finally:
        generator.close()


@pytest.mark.parametrize('kind', ['execute', 'score'])
@pytest.mark.parametrize('baseline', [False, True])
@pytest.mark.parametrize('amount', [0, 123])
def test_reservation_only_completion_reaches_normal_retry(database, kind, baseline, amount):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn, kind=kind, baseline=baseline)
        identity = reserve(conn, lease, amount=amount, funding='host' if baseline else 'miner_key')
        old = ledger(conn)
        cause = 'provider_error' if kind == 'execute' else 'judge_error'
        result = complete(conn, lease, cause)
        assert result['status'] == 'failed' and result['confirmation_attempt'] == lease['attempt'] + 1
        rows = ledger(conn)
        assert rows[:-1] == old
        receipt = rows[-1]
        assert receipt['entry_kind'] == 'recovery' and receipt['amount_microusd'] == 0
        assert receipt['entry_doc'] == dict(reason='attempt_completed_before_dispatch', reserved_microusd=old[-1]['amount_microusd'])
        for key in ('miner_hotkey', 'round_id', 'submission_id', 'run_id', 'stage',
                    'call_identity', 'provider', 'operation_id', 'funding_source'):
            assert receipt[key] == old[-1][key]
        with conn.cursor() as cur:
            cur.execute('SELECT status,result_doc,terminal_cause,output_ref FROM public.lab_arena_runs WHERE run_id=%s',
                        (lease['run_id'],))
            assert cur.fetchone() == ('failed', {'terminal_status': cause}, cause, None)
            cur.execute('SELECT status,previous_runner_hotkey FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt=%s',
                        (lease['assignment_id'], lease['attempt'] + 1))
            assert cur.fetchone() == ('pending', base.RUNNER)
        assert complete(conn, lease, cause)['idempotent'] is True
        assert ledger(conn) == rows
        assert dispatch(conn, lease, identity)['status'] == 'stale'
        assert ledger(conn) == rows


@pytest.mark.parametrize('kind', ['execute', 'score'])
@pytest.mark.parametrize('baseline', [False, True])
def test_successful_completion_releases_unused_reservation(database, kind, baseline):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn, kind=kind, baseline=baseline)
        reserve(conn, lease)
        result = complete(conn, lease)
        assert result['status'] == 'accepted' and 'confirmation_attempt' not in result
        with conn.cursor() as cur:
            cur.execute('SELECT output_ref FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone() == ('retained-output',)
        assert [row['entry_kind'] for row in ledger(conn)] == ['reservation', 'recovery']


def test_tenth_miner_judge_attempt_with_zero_scrapingdog_reservation_can_retry(database):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn, kind='score')
        with conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE kind='execute' ORDER BY run_id")
            sources = cur.fetchall()
            cur.execute('SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (base.ROUND,))
            configuration = cur.fetchone()
        for attempt in range(3, 10):
            assert lease['attempt'] == attempt
            assert complete(conn, lease, 'judge_error')['confirmation_attempt'] == attempt + 1
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            lease = score_claim(conn, str(attempt))
        assert lease['attempt'] == 10
        identity = 'sha256:' + '9' * 64
        assert base.rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'scrapingdog.search','scrapingdog','miner_key',0,'{\"action_sequence\":15}'::jsonb,4500)",
                        (lease['run_id'], TOKEN, identity))['status'] == 'reserved'
        assert complete(conn, lease, 'judge_error')['confirmation_attempt'] == 11
        assert score_claim(conn, 'backoff') == {'status': 'no_pending'}
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        successor = score_claim(conn, 'successor')
        assert (successor['status'], successor['attempt'], successor['submission_id'], successor['scored_run_id']) == (
            'leased', 11, lease['submission_id'], lease['scored_run_id'])
        assert complete(conn, successor)['status'] == 'accepted'
        with conn.cursor() as cur:
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE kind='execute' ORDER BY run_id")
            assert cur.fetchall() == sources
            cur.execute('SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (base.ROUND,))
            assert cur.fetchone() == configuration


@pytest.mark.parametrize('kind', ['execute', 'score'])
def test_legacy_nonzero_reservation_retains_original_reserved_amount(database, kind):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn, kind=kind)
        legacy_cost_policy(conn, lease)
        reserve(conn, lease, amount=123)
        old = ledger(conn)
        assert old[-1]['amount_microusd'] == 123
        assert complete(conn, lease)['status'] == 'accepted'
        assert ledger(conn)[-1]['entry_doc']['reserved_microusd'] == 123
        assert ledger(conn)[:-1] == old


@pytest.mark.parametrize('kind', ['execute', 'score'])
def test_mixed_heads_release_only_unused_reservation_and_keep_dispatch_open(database, kind):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn, kind=kind)
        unused = reserve(conn, lease, 'unused')
        inflight = reserve(conn, lease, 'inflight')
        assert dispatch(conn, lease, inflight)['status'] == 'dispatched'
        old = ledger(conn)
        assert complete(conn, lease) == {'status': 'accounting_open', 'open_calls': 1}
        rows = ledger(conn)
        assert rows[:-1] == old and rows[-1]['call_identity'] == unused
        assert rows[-1]['entry_kind'] == 'recovery'
        assert complete(conn, lease) == {'status': 'accounting_open', 'open_calls': 1}
        assert dispatch(conn, lease, unused)['status'] == 'recovered'
        assert ledger(conn) == rows
        with conn.cursor() as cur:
            cur.execute('SELECT status,result_doc FROM public.lab_arena_runs WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone() == ('leased', None)


@pytest.mark.parametrize('terminal', ['settlement', 'uncertain'])
@pytest.mark.parametrize('kind', ['execute', 'score'])
def test_existing_terminal_cost_and_evidence_are_unchanged(database, terminal, kind):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn, kind=kind)
        legacy_cost_policy(conn, lease)
        reserve(conn, lease, 'unused')
        identity = reserve(conn, lease, 'paid', amount=789)
        assert dispatch(conn, lease, identity)['status'] == 'dispatched'
        if terminal == 'settlement':
            result = base.rpc(conn, 'lab_arena_settle_call(%s,%s,%s,456,%s::jsonb,4500)',
                              (lease['run_id'], TOKEN, identity, json.dumps({'status': 200, 'call_succeeded': True})))
            assert result['status'] == 'settled'
        else:
            result = base.rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
                              (lease['run_id'], TOKEN, identity, json.dumps({'reason': 'missing_provider_cost', 'call_succeeded': True, 'provider_status': 200})))
            assert result['status'] == 'uncertain'
        old = ledger(conn)
        assert complete(conn, lease)['status'] == 'accepted'
        assert ledger(conn)[:-1] == old
        assert old[-1]['entry_kind'] == terminal
        assert old[-1]['amount_microusd'] == (456 if terminal == 'settlement' else 789)


@pytest.mark.parametrize('fault', ['wrong_token', 'expired', 'generation', 'null_cause', 'missing_status', 'wrong_kind', 'invalid_output'])
def test_rejected_completion_never_recovers_reservation(database, fault):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn)
        reserve(conn, lease)
        if fault in ('expired', 'generation'):
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET " +
                            ("lease_expires_at=clock_timestamp()-interval '1 second'" if fault == 'expired' else 'stage_generation=stage_generation-1') +
                            ' WHERE run_id=%s', (lease['run_id'],))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
        old = ledger(conn)
        if fault in ('wrong_token', 'expired', 'generation'):
            assert complete(conn, lease, token='sha256:' + 'f' * 64 if fault == 'wrong_token' else TOKEN) == {'status': 'stale'}
        else:
            with pytest.raises(database[0].Error, match='lab_arena_complete_'):
                if fault in ('missing_status', 'invalid_output'):
                    base.rpc(conn, 'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
                             (lease['run_id'], TOKEN, '{}' if fault == 'missing_status' else '{"terminal_status":"accepted"}', 'accepted', '' if fault == 'invalid_output' else 'output'))
                else:
                    complete(conn, lease, None if fault == 'null_cause' else 'judge_error')
            conn.rollback()
        assert ledger(conn) == old


@pytest.mark.parametrize('accepted', [False, True])
def test_existing_company_judgment_reservation_cleanup_follows_terminal_status(database, accepted):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn, kind='score')
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs='[]'::jsonb WHERE run_id=%s",
                        (lease['run_id'],))
            cur.execute("INSERT INTO public.lab_arena_company_judgment_reservations(cache_key,authority_slot,company_input_hash,run_id,lease_generation,runner_hotkey,lease_expires_at) SELECT 'sha256:'||repeat('1',64),0,'sha256:'||repeat('2',64),run_id,lease_generation,runner_hotkey,lease_expires_at FROM public.lab_arena_runs WHERE run_id=%s",
                        (lease['run_id'],))
            cur.execute('SELECT to_jsonb(j) FROM public.lab_arena_company_judgments j ORDER BY cache_key,authority_slot')
            immutable_judgments = cur.fetchall()
            cur.execute('SELECT to_jsonb(j) FROM public.lab_arena_judgment_cache j ORDER BY cache_key')
            immutable_cache = cur.fetchall()
        conn.commit()
        reserve(conn, lease, 'unused')
        identity = reserve(conn, lease, 'inflight')
        assert dispatch(conn, lease, identity)['status'] == 'dispatched'
        assert complete(conn, lease) == {'status': 'accounting_open', 'open_calls': 1}
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_company_judgment_reservations WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone() == (1,)
        assert base.rpc(conn, "lab_arena_settle_call(%s,%s,%s,0,'{\"call_succeeded\":false,\"status\":503}'::jsonb,4500)",
                        (lease['run_id'], TOKEN, identity))['status'] == 'settled'
        assert complete(conn, lease, 'accepted' if accepted else 'judge_error')['status'] == ('accepted' if accepted else 'failed')
        with conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_company_judgment_reservations WHERE run_id=%s', (lease['run_id'],))
            assert cur.fetchone() == (0,)
            cur.execute('SELECT to_jsonb(j) FROM public.lab_arena_company_judgments j ORDER BY cache_key,authority_slot')
            assert cur.fetchall() == immutable_judgments
            cur.execute('SELECT to_jsonb(j) FROM public.lab_arena_judgment_cache j ORDER BY cache_key')
            assert cur.fetchall() == immutable_cache


@pytest.mark.parametrize('winner', ['completion', 'dispatch'])
def test_completion_and_dispatch_serialize_on_existing_run_lock(database, winner):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = seed(conn)
        identity = reserve(conn, lease)
    owner = psycopg.connect(**dsn)
    try:
        first = complete(owner, lease, commit=False) if winner == 'completion' else dispatch(owner, lease, identity, commit=False)
        assert first['status'] == ('accepted' if winner == 'completion' else 'dispatched')
        started = threading.Event()
        def contender():
            with psycopg.connect(**dsn) as other:
                started.set()
                return dispatch(other, lease, identity) if winner == 'completion' else complete(other, lease)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(contender)
            assert started.wait(2)
            with psycopg.connect(**dsn) as observer:
                for _ in range(200):
                    with observer.cursor() as cur:
                        cur.execute("SELECT count(*) FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE 'SELECT public.lab_arena_%'")
                        if cur.fetchone()[0]:
                            break
                    threading.Event().wait(.01)
                else:
                    pytest.fail('completion/dispatch contender did not wait on the run lock')
            owner.commit()
            result = future.result(timeout=5)
        assert result == ({'status': 'stale'} if winner == 'completion' else {'status': 'accounting_open', 'open_calls': 1})
        with psycopg.connect(**dsn) as conn:
            assert [row['entry_kind'] for row in ledger(conn)] == ['reservation', 'recovery' if winner == 'completion' else 'dispatch']
    finally:
        owner.rollback()
        owner.close()
