"""Exact funding proof uses call/native keys rather than rescanning a busy run."""
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import re
import time

import pytest

from tests.lab_arena import funding_proof_throughput554_postgres_test as base
from tests.lab_arena import proven_funding_recovery524_postgres_test as prior
from tests.lab_arena import funding_proof_scalars526_postgres_test as scalar

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/558-lab-arena-funding-proof-keyed-lookups.sql'
PRE = base.POST
POST = '23fbb6e5d40450916b75110d16d9b3c9c369264014dea51e17db7f3f978786c6'


def old_source(post):
    text = SQL.read_text()
    for index in reversed(range(1, 5)):
        label = 'lookup558_' + str(index)
        old = re.search(r'\$old_' + label + r'\$(.*?)\$old_' + label + r'\$', text, re.S)[1]
        new = re.search(r'\$new_' + label + r'\$(.*?)\$new_' + label + r'\$', text, re.S)[1]
        assert post.count(new) == 1
        post = post.replace(new, old)
    assert hashlib.sha256(post.encode()).hexdigest() == PRE
    return post


@pytest.fixture(scope='module')
def database():
    gen = base.database.__wrapped__()
    db = next(gen)
    try:
        c = db[0].connect(**db[1])
        c.autocommit = True
        with c.cursor() as q:
            q.execute(SQL.read_text())
            q.execute(SQL.read_text())
            assert hashlib.sha256(base.definition(q).encode()).hexdigest() == POST
        c.close()
        yield db
    finally:
        gen.close()


test_paid_new_dispatch_retires_stop_without_mutating_history = prior.test_paid_new_dispatch_retires_stop_without_mutating_history
test_untrusted_or_stale_paid_proof_never_retires_credit = prior.test_untrusted_or_stale_paid_proof_never_retires_credit
test_existing_single_dispatch_fence_still_prevents_duplicate_paid_identity = prior.test_existing_single_dispatch_fence_still_prevents_duplicate_paid_identity
test_authorized_inherited_root_uses_current_paid_proof_but_preserves_old_proof = prior.test_authorized_inherited_root_uses_current_paid_proof_but_preserves_old_proof
test_new_denial_consumes_old_watermark_and_stops_normal_retry_again = prior.test_new_denial_consumes_old_watermark_and_stops_normal_retry_again
test_every_exhausted_key_requires_its_own_new_paid_dispatch = prior.test_every_exhausted_key_requires_its_own_new_paid_dispatch
test_record_projection_preserves_malformed_and_exact_json_guards = scalar.test_record_projection_preserves_malformed_and_exact_json_guards
test_historical_native_duplicate_keeps_exact_scope_and_latest_head = base.test_historical_native_duplicate_keeps_exact_scope_and_latest_head


def recovery_select(definition):
    start = definition.index('    WITH required AS MATERIALIZED (')
    end = definition.index('    )) FROM required k', start) + len('    )) FROM required k')
    text = definition[start:end]
    for index, name in enumerate(('round_id', 'submission_id', 'miner_hotkey', 'scoring_generation_floor'), 1):
        text = text.replace('requested_run.' + name, '$' + str(index))
    return text.replace('jsonb_array_elements(credit_keys)', 'jsonb_array_elements($5)')


def nodes(value):
    if isinstance(value, dict):
        if 'Node Type' in value:
            yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child)


def test_dense_calls_keep_exact_index_keys_and_seventy_normal_retries(database):
    pg, dsn = database
    run, denial = prior.seed(database)
    with pg.connect(**dsn) as c, c.cursor() as q:
        ids = base._queue70(c, run, denial)
        funded = prior.clone(c, run, 'dense-keyed-history')
        for index in range(1800):
            prior.call(c, funded, 'historical-native-' + str(index), at=denial-timedelta(minutes=5))
        for index in range(80):
            prior.call(c, funded, 'recent-native-' + str(index))
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_ledger SET terminal_response=terminal_response||jsonb_build_object('retained_body',repeat('wide-native-receipt-',30000)) WHERE run_id=%s AND entry_kind='settlement'", (funded,))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('CREATE INDEX IF NOT EXISTS lab_arena_ledger_cost_run_idx ON public.lab_arena_ledger(submission_id,run_id)')
        q.execute('ANALYZE public.lab_arena_ledger')
        q.execute('ANALYZE public.lab_arena_runs')
        post = base.definition(q)
        keys = json.dumps([{'provider': 'deepline', 'fingerprint': prior.FINGERPRINT, 'denied_at': denial.isoformat()}])
        parameters = (run['round_id'], run['submission_id'], run['miner_hotkey'], run['scoring_generation_floor'], keys)
        plans = {}
        for name, source in [('before', old_source(post)), ('after', post)]:
            q.execute('SET LOCAL plan_cache_mode=force_generic_plan')
            q.execute('PREPARE funding_keyed_test(text,text,text,bigint,jsonb) AS ' + recovery_select(source))
            q.execute('EXPLAIN(FORMAT JSON) EXECUTE funding_keyed_test(%s,%s,%s,%s,%s)', parameters)
            plans[name] = list(nodes(q.fetchone()[0]))
            q.execute('DEALLOCATE funding_keyed_test')
        old_native = [node for node in plans['before'] if node.get('Index Name') == 'lab_arena_ledger_paid_native_request_idx']
        new_native = [node for node in plans['after'] if node.get('Index Name') == 'lab_arena_ledger_paid_native_request_idx']
        assert old_native and not all('success.native_request_id' in node.get('Index Cond', '') for node in old_native)
        assert new_native and all('success.native_request_id' in node.get('Index Cond', '') for node in new_native)
        reservations = [node for node in plans['after'] if node.get('Index Name') == 'lab_arena_ledger_reservation_uq']
        assert len(reservations) == 2
        assert all('call_identity' in node.get('Index Cond', '') for node in reservations)
        c.commit()
        q.execute("SET LOCAL statement_timeout='8s'")
        started = time.monotonic()
        q.execute('SELECT bool_or(public.lab_arena_miner_credit_failure_v1(run_id)) FROM public.lab_arena_runs WHERE run_id=ANY(%s)', (ids,))
        assert q.fetchone() == (False,)
        proof_seconds = time.monotonic()-started
        c.commit()
        q.execute("SET LOCAL statement_timeout='8s'")
        started = time.monotonic()
        q.execute('SELECT public.lab_arena_expire_leases(%s)', (run['round_id'],))
        assert q.fetchone()[0] == {'status': 'ok', 'expired': 0, 'retried': 70}
        sweep_seconds = time.monotonic()-started
        q.execute("SELECT count(*) FROM public.lab_arena_runs WHERE run_id=ANY(%s) AND status='failed' AND terminal_cause='credential_error'", (ids,))
        assert q.fetchone() == (70,)
        print('FUNDING558_DENSE70 proof_seconds=%.3f sweep_seconds=%.3f' % (proof_seconds, sweep_seconds), flush=True)


def test_migration_replay_preserves_security_other_functions_and_rows(database):
    prior.seed(database)
    with database[0].connect(**database[1]) as c, c.cursor() as q:
        q.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
        before = q.fetchall()
        rows = prior.snapshot(q)
        q.execute(old_source(base.definition(q)))
        q.execute(SQL.read_text())
        q.execute(SQL.read_text())
        q.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
        assert q.fetchall() == before
        assert prior.snapshot(q) == rows
