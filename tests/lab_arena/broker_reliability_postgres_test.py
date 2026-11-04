"""Real PostgreSQL arbitration and durable terminal proof; no provider network."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import threading
from uuid import uuid4

import pytest

from lab_arena import broker as br
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.broker_reliability_test import SUCCESS
from tests.lab_arena.confirmed_cost_admission_postgres_test import _open_current_scoring
from tests.lab_arena.lab_arena_pg_harness import CURRENT_PROVIDER_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _start_parallel_round
from tests.lab_arena.test_lab_arena_broker import CHAT, CONTEXT, FakeTransport, make_broker
from tests.lab_arena.test_lab_arena_migration_postgres import claim, sha
from tests.lab_arena.test_lab_arena_service_round import Harness

SQL = Path(__file__).parents[2] / 'scripts/619-lab-arena-release-undispatched-call.sql'


@pytest.fixture(scope='module')
def database():
    yield from database_with_lab_arena_migration(CURRENT_PROVIDER_SERVICE_MIGRATIONS + (SQL.name,))


@pytest.fixture(params=['baseline', 'miner', 'score'])
def leased(database, tmp_path, request):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    round_id = 'arena-2099-06-01-' + uuid4().hex[:8]
    harness = None
    if request.param == 'score':
        store = ArenaStore(PsycopgTransport(connect))
        _, runners, _ = _open_current_scoring(store, round_id)
        lease, token = claim(store, round_id, runners[0], parallelism=8, ceiling=8, excluded=[runners[0]])[:2]
    else:
        harness = Harness(connect, tmp_path, challengers=['BrokerReliability'], runners=['reliability'])
        harness.service.config.defaults = replace(harness.service.config.defaults,
            per_icp_cost_policy=True, integrity_from='2000-01-01T00:00:00Z')
        participants = _start_parallel_round(harness, round_id, slot_ceiling=8)
        baseline = next(p for p in participants if harness.flavors[p['submission_id']] == 'PublicBaseline')
        target = baseline if request.param == 'baseline' else next(p for p in participants if p != baseline)
        store = harness.service.store
        lease, token = claim(store, round_id, harness.runner_keys[0], parallelism=8, ceiling=8,
            excluded=[p['miner_hotkey'] for p in participants if p != target])[:2]
    assert lease['status'] == 'leased'
    run = store.get_run(lease['run_id'])
    context = replace(CONTEXT, **{key: run[key] for key in (
        'run_id', 'assignment_id', 'icp_position', 'miner_hotkey', 'submission_id', 'stage', 'kind', 'attempt', 'round_id')},
        lease_token_hash=hash_lease_token(token))
    yield store, context, database
    store.close()


def _broker(store, context, transport):
    funding = store.provider_funding(context.run_id, 'openrouter')['funding_source']
    return make_broker(store=store, transport=transport, judge_models=(CHAT['model'],),
        funding_source_for=lambda _: funding)[0]


def _execute(broker, context, **kwargs):
    return broker.execute(context, operation_id='openrouter.chat', parameters=CHAT,
        action_sequence=0, timeout_ms=30000, **kwargs)


@pytest.mark.parametrize('failure', ['missing', 'settlement', 'first_uncertainty'])
def test_real_success_survives_new_broker_and_billing_failure(leased, monkeypatch, failure):
    store, context, _ = leased
    payload = deepcopy(SUCCESS)
    if failure == 'settlement':
        payload['usage'] = {'cost': '0.001'}
        def rejected(**kwargs):
            raise br.ArenaStoreError('settlement contract mismatch')
        monkeypatch.setattr(store, 'settle_call', rejected)
    if failure == 'first_uncertainty':
        write, attempts = store.mark_uncertain, []
        def unavailable(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise br.ArenaStoreUnavailable('database unavailable')
            return write(**kwargs)
        monkeypatch.setattr(store, 'mark_uncertain', unavailable)
    transport = FakeTransport([(200, payload)])
    first = _execute(_broker(store, context, transport), context)
    assert first.status == 200 and first.call['outcome'] == 'uncertain'
    assert first.call.get('actual_microusd') is None
    rows = store.list_ledger(call_identity=first.call['call_identity'])
    assert [r['entry_kind'] for r in rows] == ['reservation', 'dispatch', 'uncertain']
    saved = rows[-1]['entry_doc']['call']
    assert saved['openrouter_terminal_response']['call_succeeded'] is True
    assert saved.get('known_actual_microusd') == (1000 if failure == 'settlement' else None)
    # New broker instance has no process-local response cache.
    repeat = _execute(_broker(store, context, transport), context)
    assert repeat.status == 200 and repeat.body == first.body
    assert repeat.call['outcome'] == 'uncertain' and repeat.call.get('actual_microusd') is None
    assert len(transport.sent) == 1


def test_real_cancel_during_reservation_recovers_without_dispatch(leased, monkeypatch):
    store, context, _ = leased
    cancelled = [False]
    reserve = store.reserve_call
    def cancel_after_reserve(**kwargs):
        result = reserve(**kwargs)
        cancelled[0] = True
        return result
    monkeypatch.setattr(store, 'reserve_call', cancel_after_reserve)
    transport = FakeTransport([(200, SUCCESS)])
    broker = _broker(store, context, transport)
    result = _execute(broker, context, cancel_requested=lambda: cancelled[0])
    assert result.call['outcome'] == 'not_dispatched' and transport.sent == []
    rows = store.list_ledger(call_identity=result.call['call_identity'])
    assert [r['entry_kind'] for r in rows] == ['reservation', 'recovery']
    assert rows[-1]['amount_microusd'] == 0
    cancelled[0] = False
    assert _execute(broker, context).status != 200 and transport.sent == []


def _reservation(store, context):
    identity, call_doc = sha(uuid4().hex), {'request_hash': sha(uuid4().hex)}
    funding = store.provider_funding(context.run_id, 'openrouter')['funding_source']
    result = store.reserve_call(run_id=context.run_id, lease_token_hash=context.lease_token_hash,
        call_identity=identity, operation_id='openrouter.chat', provider='openrouter',
        funding_source=funding, amount_microusd=1000, call_doc=call_doc)
    assert result['status'] == 'reserved'
    return identity, call_doc


def test_real_release_checks_exact_request_and_current_lease(leased):
    store, context, _ = leased
    identity, doc = _reservation(store, context)
    arguments = dict(run_id=context.run_id, lease_token_hash=context.lease_token_hash,
        call_identity=identity, call_doc=doc, reason='provider_admission_cancelled')
    assert store.release_undispatched_call(**dict(arguments, call_doc={'request_hash': sha('changed')}))['status'] == 'reservation_mismatch'
    assert store.release_undispatched_call(**dict(arguments, lease_token_hash=sha('wrong')))['status'] == 'stale'
    assert store.release_undispatched_call(**arguments)['status'] == 'recovered'
    assert store.release_undispatched_call(**arguments)['status'] == 'recovered'
    assert store.mark_dispatched(run_id=context.run_id, lease_token_hash=context.lease_token_hash,
        call_identity=identity)['status'] == 'recovered'
    assert [r['entry_kind'] for r in store.list_ledger(call_identity=identity)] == ['reservation', 'recovery']


def test_real_release_and_dispatch_are_mutually_exclusive(leased):
    store, context, (pg, dsn) = leased
    for _ in range(4):
        identity, doc = _reservation(store, context)
        barrier = threading.Barrier(2)
        def worker(release):
            other = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
            try:
                barrier.wait(timeout=5)
                args = dict(run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=identity)
                return (other.release_undispatched_call(**args, call_doc=doc, reason='provider_admission_deadline')
                        if release else other.mark_dispatched(**args))
            finally:
                other.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, (True, False)))
        kinds = [r['entry_kind'] for r in store.list_ledger(call_identity=identity)]
        assert kinds in (['reservation', 'dispatch'], ['reservation', 'recovery'])
        assert {r['status'] for r in results} == ({'dispatched'} if kinds[-1] == 'dispatch' else {'recovered'})


def test_migration_replay_and_privileges(database):
    pg, dsn = database
    signature = 'public.lab_arena_release_undispatched_call(text,text,text,jsonb,text,bigint)'
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        before = cursor.fetchone()[0]
        cursor.execute(SQL.read_text())
        cursor.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
        assert cursor.fetchone()[0] == before
        for role in ('anon', 'authenticated', 'service_role'):
            cursor.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role, signature))
            assert cursor.fetchone()[0] is False
        cursor.execute('SELECT has_function_privilege(\'lab_arena_service\',%s,\'EXECUTE\')', (signature,))
        assert cursor.fetchone()[0] is True


def test_real_success_replays_after_delayed_generation_bill(leased):
    store, context, _ = leased
    payload = {**SUCCESS, 'id': 'gen-retained-success'}
    transport = FakeTransport([(200, payload), (404, {'error': {'code': 404}})])
    first = _execute(_broker(store, context, transport), context)
    assert first.status == 200 and first.call['outcome'] == 'uncertain'
    rows = store.list_ledger(call_identity=first.call['call_identity'])
    saved = rows[-1]['entry_doc']['call']
    result = store.reconcile_openrouter_cost(round_id=context.round_id, run_id=context.run_id,
        call_identity=first.call['call_identity'], uncertain_entry_id=rows[-1]['entry_id'],
        generation_id=saved['openrouter_generation_id'], credential_fingerprint=saved['credential_fingerprint'],
        actual_microusd=19, cost_units='0.000019')
    assert result['status'] == 'settled'
    replay = _execute(_broker(store, context, transport), context)
    assert replay.status == 200 and replay.body == first.body
    assert replay.call['outcome'] == 'settled' and replay.call['actual_microusd'] == 19
    assert [sent['method'] for sent in transport.sent].count('POST') == 1
    assert all(sent['method'] == 'GET' for sent in transport.sent[1:])


def test_only_fresh_exact_marker_allows_no_transport_recovery(leased):
    store, context, _ = leased
    identity, doc = _reservation(store, context)
    args = dict(run_id=context.run_id, lease_token_hash=context.lease_token_hash, call_identity=identity)
    fresh = store.mark_dispatched(**args)
    repeated = store.mark_dispatched(**args)
    assert fresh['idempotent'] is False and type(fresh['dispatch_entry_id']) is int
    assert repeated['idempotent'] is True and 'dispatch_entry_id' not in repeated
    release = dict(args, call_doc=doc, reason='provider_admission_cancelled')
    assert store.release_undispatched_call(**release)['status'] == 'dispatched'
    assert store.release_undispatched_call(**release, dispatch_entry_id=fresh['dispatch_entry_id'] + 1)['status'] == 'dispatched'
    assert store.release_undispatched_call(**dict(release, lease_token_hash=sha('wrong')),
        dispatch_entry_id=fresh['dispatch_entry_id'])['status'] == 'stale'
    assert store.release_undispatched_call(**release, dispatch_entry_id=fresh['dispatch_entry_id'])['status'] == 'recovered'
    rows = store.list_ledger(call_identity=identity)
    assert [r['entry_kind'] for r in rows] == ['reservation', 'dispatch', 'recovery']
    assert rows[-1]['entry_doc']['transport_not_started'] is True
    assert rows[-1]['entry_doc']['undispatched_dispatch_entry_id'] == rows[1]['entry_id']
    assert rows[-1]['amount_microusd'] == 0


def test_cancel_during_fresh_marker_recovers_exact_marker(leased, monkeypatch):
    store, context, _ = leased
    cancelled = [False]
    dispatch = store.mark_dispatched
    def cancel_after_marker(**kwargs):
        result = dispatch(**kwargs)
        cancelled[0] = True
        return result
    monkeypatch.setattr(store, 'mark_dispatched', cancel_after_marker)
    transport = FakeTransport([(200, SUCCESS)])
    first = _execute(_broker(store, context, transport), context, cancel_requested=lambda: cancelled[0])
    assert first.call['outcome'] == 'not_dispatched' and transport.sent == []
    rows = store.list_ledger(call_identity=first.call['call_identity'])
    assert [r['entry_kind'] for r in rows] == ['reservation', 'dispatch', 'recovery']
    assert rows[-1]['entry_doc']['undispatched_dispatch_entry_id'] == rows[1]['entry_id']


def test_serialized_long_settlement_outage_is_sql_reconcilable(leased, monkeypatch):
    store, context, _ = leased
    now = [0.0]
    monkeypatch.setattr(br.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(br.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    def unavailable(**kwargs):
        raise br.ArenaStoreUnavailable('database temporarily unavailable')
    monkeypatch.setattr(store, 'settle_call', unavailable)
    transport = FakeTransport([(200, {**SUCCESS, 'id': 'gen-long-outage', 'usage': {'cost': '0.001'}})])
    first = _execute(_broker(store, context, transport), context)
    assert first.status == 200 and first.call['outcome'] == 'uncertain'
    assert now[0] == 120 and first.call.get('actual_microusd') is None
    rows = store.list_ledger(call_identity=first.call['call_identity'])
    saved = rows[-1]['entry_doc']['call']
    assert saved['error_class'] == 'ArenaStoreError' and saved['reason'] == 'settle_failure'
    assert saved['failure_stage'] == 'settlement' and saved['known_actual_microusd'] == 1000
    candidates = store.list_openrouter_cost_reconciliations(context.round_id, run_id=context.run_id, limit=20)
    assert [r['call_identity'] for r in candidates] == [first.call['call_identity']]
    result = store.reconcile_openrouter_cost(round_id=context.round_id, run_id=context.run_id,
        call_identity=first.call['call_identity'], uncertain_entry_id=rows[-1]['entry_id'],
        generation_id=saved['openrouter_generation_id'], credential_fingerprint=saved['credential_fingerprint'],
        actual_microusd=1000, cost_units='0.001')
    assert result['status'] == 'settled'
    repeat = _execute(_broker(store, context, transport), context)
    assert repeat.status == 200 and repeat.body == first.body
    assert repeat.call['actual_microusd'] == 1000 and len(transport.sent) == 1


def test_migration_runs_without_superuser_and_restores_schema_acl(database):
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        # Reconstruct the exact pre-619 function and remove only this migration's
        # new RPC. This exercises first CREATE/ALTER OWNER, not just replay.
        base_sql = (SQL.parent / '179-lab-arena-v1.sql').read_text()
        start = base_sql.index('CREATE OR REPLACE FUNCTION public.lab_arena_mark_dispatched(')
        end = base_sql.index('-- Settle a dispatched call', start)
        cursor.execute(base_sql[start:end])
        signature = 'public.lab_arena_release_undispatched_call(text,text,text,jsonb,text,bigint)'
        cursor.execute('DROP FUNCTION ' + signature)
        cursor.execute("""SELECT encode(extensions.digest(pg_get_functiondef(
            'public.lab_arena_mark_dispatched(text,text,text)'::regprocedure),'sha256'),'hex')""")
        assert cursor.fetchone()[0] == '3149e4a6f481ca8dee40831b811674f7c540936d8d73ae346f55b0a1eea4f12f'
        cursor.execute('SELECT to_regprocedure(%s)', (signature,))
        assert cursor.fetchone()[0] is None
        cursor.execute('CREATE ROLE release619_migrator NOLOGIN')
        cursor.execute('GRANT lab_arena_owner TO release619_migrator')
        cursor.execute('ALTER SCHEMA public OWNER TO release619_migrator')
        cursor.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
        cursor.execute("SELECT nspacl FROM pg_namespace WHERE nspname='public'")
        before = cursor.fetchone()[0]
        cursor.execute('SET ROLE release619_migrator')
        cursor.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert cursor.fetchone()[0] is False
        cursor.execute(SQL.read_text())
        cursor.execute('SELECT to_regprocedure(%s)', (signature,))
        assert cursor.fetchone()[0] is not None
        cursor.execute("SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') FROM pg_namespace WHERE nspname='public'")
        after, owner_create = cursor.fetchone()
        assert after == before and owner_create is False
        cursor.execute('RESET ROLE')
