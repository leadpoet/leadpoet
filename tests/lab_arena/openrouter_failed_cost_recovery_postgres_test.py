"""The real ledger retains error replay and settles a delayed exact bill once."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lab_arena import broker as br, training_export
from lab_arena.store import hash_lease_token
from scripts.verify_arena_full_thinking_e2e import full_thinking_database
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _start_parallel_round
from tests.lab_arena.test_lab_arena_migration_postgres import claim
from tests.lab_arena.test_lab_arena_service_round import Harness
from tests.lab_arena.test_lab_arena_broker import CHAT, FakeTransport, HOST_KEYS, price_table


@pytest.fixture(scope='module')
def database():
    fixture = full_thinking_database()
    try:
        db = next(fixture)
        connection = db[0].connect(**db[1])
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                for migration in (
                    '575-lab-arena-score-host-continuation.sql',
                    '576-lab-arena-champion-score-funding.sql',
                    '577-lab-arena-judge-continuation-runner-handoff.sql',
                    '578-lab-arena-closed-provider-ordered-candidate.sql',
                    '579-lab-arena-close-covered-credit-candidates.sql',
                ):
                    cursor.execute((Path(__file__).resolve().parents[2] / 'scripts' / migration).read_text())
        finally:
            connection.close()
        yield db
    finally:
        fixture.close()


def current_round(database, tmp_path, label):
    pg, dsn = database
    harness = Harness(lambda: pg.connect(**dsn), tmp_path, challengers=[], runners=['alpha'])
    harness.service.config.defaults = replace(harness.service.config.defaults,
        per_icp_cost_policy=True, flat_per_icp_cost_policy=True, integrity_from='2000-01-01T00:00:00Z')
    suffix = hashlib.sha256(label.encode()).hexdigest()[:8]
    round_id = 'arena-2099-03-01-' + suffix
    _start_parallel_round(harness, round_id, slot_ceiling=2)
    store = harness.service.store
    lease, token, _, _ = claim(store, round_id, harness.runner_keys[0], parallelism=2, ceiling=2)
    run = store.get_run(lease['run_id'])
    return store, round_id, run['submission_id'], run, hash_lease_token(token)


@pytest.mark.parametrize('operation', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('provider_status', [404, 429])
@pytest.mark.parametrize('cost', [0, 19])
def test_failed_reply_reconciles_once_and_keeps_reply(database, tmp_path, operation, provider_status, cost):
    label = 'err%s%d%d' % (operation[-1], provider_status, cost)
    store, round_id, _, run, token_hash = current_round(database, tmp_path, label)
    try:
        context = br.RunContext(run_id=run['run_id'], assignment_id=run['assignment_id'],
            icp_position=run['icp_position'], lease_token_hash=token_hash,
            miner_hotkey=run['miner_hotkey'], submission_id=run['submission_id'],
            stage=run['stage'], round_id=round_id, kind='execute', attempt=run['attempt'])
        generation = 'gen-' + label
        payload = {'error': {'code': provider_status, 'message': 'provider failed'}}
        transport = FakeTransport([(provider_status, payload, {'X-Generation-Id': generation}),
            (404, {'error': {'code': 404}}),
            (200, {'data': {'id': generation, 'total_cost': format(cost / 1_000_000, '.6f')}})])
        instance = br.Broker(store=store, key_for=lambda provider: HOST_KEYS[provider],
            price_table=price_table(), transport=transport, lease_ttl_seconds=120)
        parameters = CHAT if operation == 'openrouter.chat' else {
            'model': CHAT['model'], 'input': 'hello', 'max_output_tokens': 200}
        def execute():
            return instance.execute(context, operation_id=operation, parameters=parameters,
                action_sequence=0, timeout_ms=30000)
        first = execute()
        assert first.call['outcome'] == 'uncertain'
        assert first.call['reserved_microusd'] == 0
        replay = execute()
        assert (replay.status, replay.headers, replay.body) == (first.status, first.headers, first.body)
        candidates = store.list_openrouter_cost_reconciliations(round_id, run_id=run['run_id'], limit=10)
        assert len(candidates) == 1 and candidates[0]['generation_id'] == generation
        assert instance.reconcile_openrouter_cost(dict(candidates[0],
            credential_fingerprint='sha256:' + 'a' * 64))['status'] == 'credential_mismatch'
        wrong_generation = store.reconcile_openrouter_cost(round_id=round_id, run_id=run['run_id'],
            call_identity=first.call['call_identity'], uncertain_entry_id=candidates[0]['uncertain_entry_id'],
            generation_id='gen-another-call', credential_fingerprint=candidates[0]['credential_fingerprint'],
            actual_microusd=cost, cost_units=format(cost / 1_000_000, '.6f'))
        assert wrong_generation['status'] == 'stale'
        settled = instance.reconcile_openrouter_cost(candidates[0])
        assert settled['status'] == 'settled'
        final = execute()
        assert final.call['actual_microusd'] == cost
        assert (final.status, final.headers, final.body) == (first.status, first.headers, first.body)
        # The SQL operation itself is idempotent even when a billing reply is lost.
        again = store.reconcile_openrouter_cost(round_id=round_id, run_id=run['run_id'],
            call_identity=first.call['call_identity'], uncertain_entry_id=candidates[0]['uncertain_entry_id'],
            generation_id=generation, credential_fingerprint=candidates[0]['credential_fingerprint'],
            actual_microusd=cost, cost_units=format(cost / 1_000_000, '.6f'))
        assert again['status'] == 'settled' and again['idempotent'] is True
        rows = store.list_ledger(call_identity=first.call['call_identity'])
        assert sum(row['entry_kind'] == 'settlement' for row in rows) == 1
        exported = training_export._calls(rows, {run['run_id']}, None, {run['run_id']: 0})[0][0]
        assert exported['actual_microusd'] == cost
        assert exported['response_status'] == first.status and exported['response'] == json.loads(first.body)
        assert training_export.response_entry_id(exported) == candidates[0]['uncertain_entry_id']
        assert [row['amount_microusd'] for row in rows if row['entry_kind'] == 'reservation'] == [0]
        assert [item['method'] for item in transport.sent] == ['POST', 'GET', 'GET']
        assert store.list_openrouter_cost_reconciliations(round_id, run_id=run['run_id'], limit=10) == []
    finally:
        store.close()


@pytest.mark.parametrize('size', ['large', 'boundary', 'overlimit'])
def test_installed_uncertainty_function_and_large_error_replay(database, tmp_path, size):
    pg, dsn = database
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_get_functiondef('public.lab_arena_mark_uncertain(text,text,text,jsonb,integer)'::regprocedure)")
        definition = cursor.fetchone()[0]
    (tmp_path / 'installed-mark-uncertain.sql').write_text(definition)
    # This RPC never had the reservation RPC's 64KiB input cap. No migration
    # or new object-store callback is needed to retain the existing terminal.
    assert '65536' not in definition and 'octet_length' not in definition
    store, round_id, _, run, token_hash = current_round(database, tmp_path, 'largeerror' + size)
    try:
        context = br.RunContext(run_id=run['run_id'], assignment_id=run['assignment_id'],
            icp_position=run['icp_position'], lease_token_hash=token_hash,
            miner_hotkey=run['miner_hotkey'], submission_id=run['submission_id'],
            stage=run['stage'], round_id=round_id, kind='execute', attempt=run['attempt'])
        limit = br.operations.OPERATIONS['openrouter.chat'].max_response_bytes
        overhead = len(json.dumps({'error': {'code': 404, 'message': ''}}).encode())
        length = 900_000 if size == 'large' else limit - overhead + (size == 'overlimit')
        payload = {'error': {'code': 404, 'message': 'x' * length}}
        transport = FakeTransport([(404, payload, {'X-Generation-Id': 'gen-large-error'}),
            (404, {'error': {'code': 404}})])
        instance = br.Broker(store=store, key_for=lambda provider: HOST_KEYS[provider],
            price_table=price_table(), transport=transport, lease_ttl_seconds=120)
        first = instance.execute(context, operation_id='openrouter.chat', parameters=CHAT,
            action_sequence=0, timeout_ms=30000)
        assert first.call['outcome'] == 'uncertain'
        replay = instance.execute(context, operation_id='openrouter.chat', parameters=CHAT,
            action_sequence=0, timeout_ms=30000)
        if size == 'overlimit':
            assert first.status == 502 and replay.call['error_code'] == 'call_uncertain'
            assert first.call.get('actual_microusd') is None
            assert [item['method'] for item in transport.sent] == ['POST', 'GET']
            return  # Existing inadmissible-response behavior stays fail-closed.
        assert first.status == 404
        assert (replay.status, replay.headers, replay.body) == (first.status, first.headers, first.body)
        rows = store.list_ledger(call_identity=first.call['call_identity'])
        saved = next(row['entry_doc']['call'] for row in rows if row['entry_kind'] == 'uncertain')
        encoded = json.dumps(saved).encode()
        terminal = saved['openrouter_terminal_response']
        assert 65_536 < len(encoded) <= 4 * 1024 * 1024
        assert len(json.dumps(terminal).encode()) <= 4 * 1024 * 1024
        (tmp_path / 'large-error-proof.json').write_text(json.dumps({
            'function_sha256': hashlib.sha256(definition.encode()).hexdigest(),
            'retained_call_doc_bytes': len(encoded), 'error_body_bytes': len(first.body),
            'response_sha256': hashlib.sha256(first.body).hexdigest(),
            'replay_equal': replay.body == first.body, 'provider_methods': [v['method'] for v in transport.sent],
        }, indent=2))
        assert [item['method'] for item in transport.sent] == ['POST', 'GET']
    finally:
        store.close()
