"""Exact late OpenRouter judge bills alter only the audit ledger."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from lab_arena import broker as br, contracts
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import _open_scoring, _store
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_broker import price_table
from tests.lab_arena.test_lab_arena_migration_postgres import claim, sha

MIGRATION = Path(__file__).resolve().parents[2] / 'scripts/413-lab-arena-closed-openrouter-judge-billing.sql'
SECRET = 'synthetic-billing-fixture'


@pytest.fixture(scope='module')
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS + (
        '264-lab-arena-codex-cost-reconciliation.sql',
        '311-lab-arena-per-icp-closed-billing-reconciliation.sql', MIGRATION.name,
    ))


def _call(database, label, *, round_status='published', run_status='accepted',
          kind='score', reason='missing_provider_cost', patch=None):
    psycopg2, dsn = database
    store = _store(database)
    rid = 'arena-2026-09-14-or' + label
    runners, _ = _open_scoring(store, rid, participants=2, runners=2)
    run, token, _, _ = claim(store, rid, runners[0], parallelism=8, ceiling=8)
    assert run['kind'] == 'score'
    token_hash = hash_lease_token(token)
    identity = contracts.provider_call_identity(attempt=run['attempt'],
        assignment_id=run['assignment_id'], icp_position=run['icp_position'],
        action_sequence=0, operation_id='openrouter.chat', request_hash=sha(label))
    assert store.reserve_call(run_id=run['run_id'], lease_token_hash=token_hash,
        call_identity=identity, operation_id='openrouter.chat', provider='openrouter',
        funding_source='miner_key', amount_microusd=0, call_doc={'model': 'fixture'})['status'] == 'reserved'
    assert store.mark_dispatched(run_id=run['run_id'], lease_token_hash=token_hash,
        call_identity=identity)['status'] == 'dispatched'
    doc = {'reason': reason, 'provider_status': 200, 'call_succeeded': True,
        'openrouter_generation_id': 'gen-' + label,
        'credential_fingerprint': br._credential_fingerprint(SECRET)}
    if reason == 'settle_failure':
        doc.update(failure_stage='settlement', error_class='ArenaStoreError', known_actual_microusd=13)
    doc.update(patch or {})
    assert store.mark_uncertain(run_id=run['run_id'], lease_token_hash=token_hash,
        call_identity=identity, call_doc=doc)['status'] == 'uncertain'
    candidate = store.list_openrouter_cost_reconciliations(rid)[0] if not patch else None
    assert store.cancel_round(rid, 'test_cancel')['status'] == 'cancelled'
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute("UPDATE public.lab_arena_runs SET status=%s, kind=%s, terminal_cause=%s WHERE run_id=%s",
            (run_status, kind, 'accepted' if run_status == 'accepted' else 'judge_error', run['run_id']))
        cursor.execute("UPDATE public.lab_arena_rounds SET status=%s, publication_doc='{}'::jsonb, "
            "configuration_doc=configuration_doc || jsonb_build_object('mode','live','network_name','finney',"
            "'netuid',71,'sourcing_cost_eligibility_policy','successful_calls_per_icp_v1') WHERE round_id=%s",
            (round_status, rid))
    return store, rid, run, identity, candidate


class BillingTransport:
    def __init__(self, generation, status=200):
        self.generation, self.status, self.calls = generation, status, []

    def send(self, **kwargs):
        assert kwargs['method'] == 'GET' and not kwargs['body']
        self.calls.append(kwargs['method'])
        return br.ProviderResponse(self.status, {}, json.dumps({
            'data': {'id': self.generation, 'total_cost': '0.0000123'}
        }).encode())


def _broker(store, transport, secret=SECRET):
    return br.Broker(store=store, key_for=lambda _p: secret,
        price_table=price_table(), transport=transport,
        credential_for=lambda _c, _p: secret,
        provider_funding_source_for=lambda _c, _p: 'miner_key')


def _hashes(database, rid):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r WHERE round_id=%s", (rid,))
        round_hash = cursor.fetchone()[0]
        cursor.execute("SELECT md5(string_agg(to_jsonb(r)::text, ',' ORDER BY run_id)) FROM public.lab_arena_runs r WHERE round_id=%s", (rid,))
        return round_hash, cursor.fetchone()[0]


@pytest.mark.parametrize('reason', ['missing_provider_cost', 'settle_failure'])
@pytest.mark.parametrize('run_status', ['accepted', 'failed'])
@pytest.mark.parametrize('round_status', ['published', 'cancelled'])
def test_exact_closed_judge_bill_settles_once_without_changing_accepted_state(database, reason, run_status, round_status):
    label = reason[:2] + run_status[:2] + round_status[:2]
    store, rid, run, identity, candidate = _call(database, label, reason=reason,
        run_status=run_status, round_status=round_status)
    try:
        before = _hashes(database, rid)
        old = store.list_ledger(call_identity=identity)
        selection = store.next_closed_provider_reconciliation(mode='live', network_name='finney', netuid=71, round_id=rid)
        assert selection['provider'] == 'openrouter'
        assert store.next_closed_deepline_reconciliation(mode='live', network_name='finney', netuid=71, round_id=rid) == {'status': 'none'}
        transport = BillingTransport(candidate['generation_id'])
        broker = _broker(store, transport)
        first = broker.reconcile_openrouter_cost(candidate)
        assert first['status'] == 'settled' and first['actual_microusd'] == 13
        assert broker.reconcile_openrouter_cost(candidate) == dict(first, idempotent=True)
        assert transport.calls == ['GET', 'GET']
        new = store.list_ledger(call_identity=identity)
        assert new[:-1] == old and new[-1]['amount_microusd'] == 13
        assert _hashes(database, rid) == before
        assert store.next_closed_provider_reconciliation(mode='live', network_name='finney', netuid=71, round_id=rid) == {'status': 'none'}
    finally:
        store.close()


@pytest.mark.parametrize('patch', [
    {'openrouter_generation_id': ''}, {'credential_fingerprint': ''},
    {'failure_stage': 'reservation'}, {'error_class': 'OperationError'},
])
def test_unbound_settlement_failure_cannot_enter_closed_recovery(database, patch):
    label = 'bad' + sha(next(iter(patch)))[7:15]
    store, rid, run, identity, _ = _call(database, label, reason='settle_failure', patch=patch)
    try:
        assert store.list_openrouter_cost_reconciliations(rid) == []
        assert store.next_closed_provider_reconciliation(mode='live', network_name='finney', netuid=71, round_id=rid) == {'status': 'none'}
    finally:
        store.close()


def test_failed_lookup_and_rotated_credentials_never_invent_or_release_cost(database):
    store, rid, _, identity, candidate = _call(database, 'lookupfail')
    try:
        old = store.list_ledger(call_identity=identity)
        transport = BillingTransport(candidate['generation_id'], status=404)
        assert _broker(store, transport).reconcile_openrouter_cost(candidate) == {'status': 'pending'}
        assert _broker(store, transport, secret='rotated-fixture').reconcile_openrouter_cost(candidate) == {'status': 'credential_mismatch'}
        assert transport.calls == ['GET'] and store.list_ledger(call_identity=identity) == old
    finally:
        store.close()


def test_closed_execute_and_wrong_chain_are_excluded_and_cursor_wraps(database):
    store, rid, _, identity, candidate = _call(database, 'executecontrol', kind='execute')
    try:
        assert store.list_openrouter_cost_reconciliations(rid) == []
        assert _broker(store, BillingTransport(candidate['generation_id'])).reconcile_openrouter_cost(candidate) == {'status': 'stale'}
    finally:
        store.close()
    store, rid, _, _, candidate = _call(database, 'faircursor')
    try:
        expected = store.next_closed_provider_reconciliation(mode='live', network_name='finney', netuid=71, round_id=rid)
        assert store.next_closed_provider_reconciliation(mode='live', network_name='finney', netuid=71,
            round_id=rid, after_entry_id=candidate['uncertain_entry_id']) == expected
        assert store.next_closed_provider_reconciliation(mode='live', network_name='test', netuid=71, round_id=rid) == {'status': 'none'}
    finally:
        store.close()


def test_migration_repeats_preserves_old_selector_and_restricts_rpc(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT md5(pg_get_functiondef('public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)'::regprocedure)), nspacl::text FROM pg_namespace WHERE nspname='public'")
        before = cursor.fetchone()
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT md5(pg_get_functiondef('public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)'::regprocedure)), nspacl::text FROM pg_namespace WHERE nspname='public'")
        assert cursor.fetchone() == before
        for role, expected in [('anon', False), ('authenticated', False), ('service_role', False), ('lab_arena_service', True)]:
            cursor.execute("SELECT has_function_privilege(%s, 'public.lab_arena_next_closed_provider_reconciliation_v1(text,text,integer,text,bigint)', 'EXECUTE')", (role,))
            assert cursor.fetchone() == (expected,)
