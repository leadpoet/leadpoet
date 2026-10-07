"""Real gateway/lease and append-only response plus exact charge recovery."""
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lab_arena import broker as br
from lab_arena.store import ArenaStoreError
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import (
    frame, setup,
)
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import database as base_database
from tests.lab_arena.deepline_late_response_recovery_test import LateTransport
from tests.lab_arena.deepline_completed_response_recovery_test import NATIVE, catalog

MIGRATION = '417-lab-arena-deepline-response-recovery.sql'


@pytest.fixture(scope='module')
def database(base_database):
    with base_database[0].connect(**base_database[1]) as connection, connection.cursor() as cur:
        for filename in ['365-lab-arena-trajectories.sql', '366-lab-arena-trajectory-capacity.sql',
                         MIGRATION, MIGRATION]:
            cur.execute((Path(__file__).parents[2] / 'scripts' / filename).read_text())
    return base_database


def deliver(h, lease, token):
    return h.service.handle_provider(lease['run_id'], token, frame())


def recover_arguments(h, lease, first, broker, connect):
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT entry_doc FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='reservation'", (first['call']['call_identity'],))
        doc = cur.fetchone()[0]
        cur.execute("SELECT lease_token_hash FROM public.lab_arena_runs WHERE run_id=%s", (lease['run_id'],))
        lease_hash = cur.fetchone()[0]
    return dict(run_id=lease['run_id'], lease_token_hash=lease_hash,
        call_identity=first['call']['call_identity'], request_hash=doc['request_hash'],
        execution_key=doc['deepline_execution_key'], credential_fingerprint=doc['credential_fingerprint'],
        request_id=NATIVE, operation=doc['tool'], actual_microusd=2000,
        terminal_response=br._terminal_response_document(200, {'content-type': 'application/json'},
            b'{"result":"recovered"}', call_succeeded=True, provider_cost={
                'basis': 'deepline_credits_x_0.10_usd', 'units': '0.02', 'unit_name': 'credits',
                'operation': doc['tool'], 'request_id': NATIVE}), lease_ttl_seconds=420)


@pytest.mark.parametrize('billing_first', [False, True])
def test_late_result_and_charge_both_orders_preserve_history_and_success_cost(database, tmp_path, monkeypatch, billing_first):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = LateTransport()
    label = '12' if billing_first else '13'
    h, lease, token, connect, broker = setup(database, tmp_path, label, transport)
    first = deliver(h, lease, token)
    assert first['status'] == 502 and first['call']['outcome'] == 'uncertain'
    transport.complete = True; transport.bill = True
    if billing_first:
        candidates = h.service.store.list_deepline_cost_reconciliations(h.round_id, run_id=lease['run_id'])
        assert broker.reconcile_deepline_cost(candidates[0])['status'] == 'settled'
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT entry_id,entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id", (first['call']['call_identity'],))
        before = cur.fetchall()
    result = deliver(h, lease, token)
    assert result['status'] == 200 and result['call']['actual_microusd'] == 2000
    assert json.loads(base64.b64decode(result['body_b64']))['result']['data']['results'][0]['url'] == 'https://example.com'
    before_requests = len(transport.requests)
    replay = deliver(h, lease, token)
    assert replay['body_b64'] == result['body_b64'] and len(transport.requests) == before_requests
    assert transport.paid == 1
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT entry_id,entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id", (first['call']['call_identity'],))
        after = cur.fetchall()
        assert after[:len(before)] == before
        assert len([r for r in after if r[1] == 'settlement']) == 1
        cur.execute("SELECT terminal_response FROM public.lab_arena_deepline_call_responses WHERE call_identity=%s", (first['call']['call_identity'],))
        saved = cur.fetchone()[0]
        assert saved['body_b64'] == result['body_b64'] and saved['call_succeeded'] is True
        cur.execute("SELECT public.lab_arena__successful_icp_cost_state(%s,%s,%s)", (h.round_id, lease['submission_id'], lease['icp_position']))
        cost = cur.fetchone()[0]
        assert cost['settled_microusd'] == 2000 and cost['successful_calls'] == 1
        assert transport.key not in json.dumps(saved)


def test_matching_running_key_is_resumed_once_by_gateway(database, tmp_path, monkeypatch):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = LateTransport()
    h, lease, token, connect, broker = setup(database, tmp_path, '14', transport)
    first = deliver(h, lease, token)
    assert first['status'] == 502
    result = deliver(h, lease, token)
    assert result['status'] == 200 and transport.paid == 1
    assert sum(r['method'] == 'POST' for r in transport.requests) == 2
    h.service.store.deepline_response_schema()
    assert h.service.store.deepline_catalog_schema()['version'] == 415


@pytest.mark.parametrize('patch', ['run', 'request_hash', 'key', 'credential', 'operation', 'cost', 'request', 'stale'])
def test_response_rpc_rejects_wrong_identity_or_expired_lease(database, tmp_path, monkeypatch, patch):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = LateTransport()
    # Each case gets separate test temporary objects in a fixture round date.
    labels = {'run':'01','request_hash':'02','key':'03','credential':'04','operation':'05','cost':'06','request':'07','stale':'08'}
    h, lease, token, connect, broker = setup(database, tmp_path, labels[patch], transport)
    first = deliver(h, lease, token)
    arguments = recover_arguments(h, lease, first, broker, connect)
    if patch == 'run': arguments['run_id'] = 'foreign-run'
    elif patch == 'request_hash': arguments['request_hash'] = 'sha256:' + 'b'*64
    elif patch == 'key': arguments['execution_key'] = 'arena:' + 'b'*64
    elif patch == 'credential': arguments['credential_fingerprint'] = 'sha256:' + 'b'*64
    elif patch == 'operation': arguments['operation'] = 'parallel_other'
    elif patch == 'cost': arguments['terminal_response']['provider_cost']['request_id'] = 'wrong-native'
    elif patch == 'request': arguments['request_id'] = 'wrong-native'
    else: arguments['lease_token_hash'] = 'sha256:' + 'b'*64
    try:
        result = h.service.store.recover_deepline_response(**arguments)
        assert result['status'] in ('stale', 'conflict')
    except ArenaStoreError:
        pass
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.lab_arena_deepline_call_responses WHERE call_identity=%s", (first['call']['call_identity'],))
        assert cur.fetchone()[0] == 0


def test_concurrent_same_result_rpc_preserves_one_response_and_charge(database, tmp_path, monkeypatch):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    h, lease, token, connect, broker = setup(database, tmp_path, '09', LateTransport())
    first = deliver(h, lease, token)
    arguments = recover_arguments(h, lease, first, broker, connect)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: h.service.store.recover_deepline_response(**arguments), range(2)))
    assert all(r['status'] == 'settled' for r in results)
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.lab_arena_deepline_call_responses WHERE call_identity=%s", (first['call']['call_identity'],))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'", (first['call']['call_identity'],))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT pg_catalog.has_table_privilege('service_role','public.lab_arena_deepline_call_responses','SELECT'), pg_catalog.has_function_privilege('service_role','public.lab_arena_recover_deepline_response_v1(text,text,text,text,text,text,text,text,bigint,jsonb,integer)','EXECUTE')")
        assert cur.fetchone() == (False, False)
        with pytest.raises(Exception):
            cur.execute("UPDATE public.lab_arena_deepline_call_responses SET request_id='rewrite' WHERE call_identity=%s", (first['call']['call_identity'],))


def test_saved_unpriced_success_then_background_bill_has_exact_successful_spend(database, tmp_path, monkeypatch):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = LateTransport()
    h, lease, token, connect, broker = setup(database, tmp_path, '10', transport)
    first = deliver(h, lease, token)
    transport.complete = True
    response = deliver(h, lease, token)
    assert response['status'] == 200 and response['call']['outcome'] == 'uncertain'
    assert 'actual_microusd' not in response['call']
    before = len(transport.requests)
    assert deliver(h, lease, token)['body_b64'] == response['body_b64']
    assert len(transport.requests) == before
    candidate = h.service.store.list_deepline_cost_reconciliations(h.round_id, run_id=lease['run_id'])[0]
    transport.bill = True
    assert broker.reconcile_deepline_cost(candidate)['status'] == 'settled'
    settled = deliver(h, lease, token)
    assert settled['body_b64'] == response['body_b64'] and settled['call']['actual_microusd'] == 2000
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT public.lab_arena__successful_icp_cost_state(%s,%s,%s)", (h.round_id, lease['submission_id'], lease['icp_position']))
        state = cur.fetchone()[0]
        assert state['settled_microusd'] == 2000 and state['successful_calls'] == 1
        cur.execute("SELECT terminal_response FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'", (first['call']['call_identity'],))
        assert cur.fetchone()[0]['call_succeeded'] is False  # Original billing-only evidence stays intact.


def test_expired_lease_cannot_attach_result_or_change_original_charge(database, tmp_path, monkeypatch):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    h, lease, token, connect, broker = setup(database, tmp_path, '11', LateTransport())
    first = deliver(h, lease, token)
    arguments = recover_arguments(h, lease, first, broker, connect)
    with connect() as connection, connection.cursor() as cur:
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=pg_catalog.clock_timestamp()-INTERVAL '1 minute' WHERE run_id=%s", (lease['run_id'],))
    assert h.service.store.recover_deepline_response(**arguments)['status'] == 'stale'
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.lab_arena_deepline_call_responses WHERE call_identity=%s", (first['call']['call_identity'],))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'", (first['call']['call_identity'],))
        assert cur.fetchone()[0] == 0


def test_worker_socket_to_gateway_and_real_ledger_recovers_original_action(database, tmp_path, monkeypatch):
    import httpx
    import uuid
    from lab_arena import runner
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = LateTransport()
    h, lease, token, connect, broker = setup(database, tmp_path, '15', transport)
    class GatewayApi:
        def __init__(self):
            self.frames = []
        def provider(self, run_id, lease_token, document):
            self.frames.append(dict(document))
            return h.service.handle_provider(run_id, lease_token, document)
    api = GatewayApi()
    path = Path('/private/tmp') / ('deepline-worker-' + uuid.uuid4().hex + '.sock')
    state = runner.RunState(lease=dict(lease, deepline_catalog=catalog()), lease_token=token)
    worker = runner.WorkerSocketServer(path, api, state)
    worker.start()
    try:
        # Use the dedicated transport so other suites' client shim patches
        # and proxy environment cannot redirect this local Unix-socket probe.
        with httpx.HTTPTransport(uds=str(path)) as http_transport:
            response = http_transport.handle_request(httpx.Request('POST',
                'http://code.deepline.com/api/v2/integrations/parallel_search/execute',
                json={'payload': frame()['parameters']['payload']},
                extensions={'timeout': {'connect': 10, 'read': 10, 'write': 10, 'pool': 10}}))
            response.read()
        assert response.status_code == 200, (response.text, len(api.frames))
        assert response.json()['result']['data']['results'][0]['url'] == 'https://example.com'
        assert response.headers[runner.SETTLED_MICROUSD_HEADER] == '2000'
        assert [f['action_sequence'] for f in api.frames] == [0, 0]
        assert transport.paid == 1 and state.action_sequence == 1 and len(state.calls) == 1
        with connect() as connection, connection.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='settlement'", (lease['run_id'],))
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT COUNT(*) FROM public.lab_arena_deepline_call_responses WHERE run_id=%s", (lease['run_id'],))
            assert cur.fetchone()[0] == 1
    finally:
        worker.stop()
