"""Recovered provider output through the gateway handler, real lease, ledger and recovery RPC."""
import base64
import json
from pathlib import Path

import pytest

from lab_arena import broker as br
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import database as base_database, fixture_configuration, run
from tests.lab_arena.deepline_completed_response_recovery_test import RecoveryTransport, NATIVE, catalog
from tests.lab_arena.test_lab_arena_broker import HOST_KEYS, price_table


@pytest.fixture(scope="module")
def database(base_database):
    with base_database[0].connect(**base_database[1]) as connection, connection.cursor() as cur:
        for filename in ('365-lab-arena-trajectories.sql', '366-lab-arena-trajectory-capacity.sql'):
            cur.execute((Path(__file__).parents[2] / 'scripts' / filename).read_text())
    return base_database


def frame():
    return {'operation_id': 'deepline.execute', 'parameters': {'tool': 'parallel_search',
        'payload': {'objective': 'Find the official example.com website'}}, 'action_sequence': 0, 'timeout_ms': 1000}


def setup(database, tmp_path, label, transport):
    h, lease, token, connect = run(database, tmp_path, label, catalog=False)
    with fixture_configuration(connect) as cur:
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc = "
            "jsonb_set(jsonb_set(configuration_doc,'{call_quotas,deepline}','0'),'{deepline_catalog}',%s::jsonb) "
            "WHERE round_id=%s", (json.dumps(catalog()), h.round_id))
    broker = br.Broker(store=h.service.store, key_for=lambda provider: HOST_KEYS[provider],
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: h.service.store.provider_funding(lease['run_id'], 'deepline')['funding_source'],
        price_table=price_table(), transport=transport, clock=h.clock)
    h.service._brokers[h.round_id] = broker
    return h, lease, token, connect, broker


def test_saved_response_returns_research_and_settles_real_lease_once(database, tmp_path):
    transport = RecoveryTransport(lost_error=br.ProviderTransportError('ReadTimeout', observed_status=200,
        deepline_job_id='iad1::trace-1791330233666-aeb0e7021ee6'))
    h, lease, token, connect, broker = setup(database, tmp_path, '17', transport)
    result = h.service.handle_provider(lease['run_id'], token, frame())
    assert result['status'] == 200 and result['call']['actual_microusd'] == 2000
    assert json.loads(base64.b64decode(result['body_b64']))['result']['data']['results'][0]['url'] == 'https://example.com'
    before = len(transport.requests)
    replayed = h.service.handle_provider(lease['run_id'], token, frame())
    assert replayed['call']['idempotent'] is True and len(transport.requests) == before
    assert sum(r['method'] == 'POST' for r in transport.requests) == 1
    assert h.service.store.list_deepline_cost_reconciliations(h.round_id, run_id=lease['run_id']) == []
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger "
                    "WHERE call_identity=%s ORDER BY entry_id", (result['call']['call_identity'],))
        entries = cur.fetchall()
        cur.execute("SELECT content->'call'->>'transport_recovery' FROM public.lab_arena_trajectory_events "
                    "WHERE run_id=%s AND event_kind='provider.response' ORDER BY trajectory_id LIMIT 1", (lease['run_id'],))
        assert cur.fetchone()[0] == 'deepline_execution_lookup'
    assert entries[0][0:2] == ('reservation', 0)
    settlements = [r for r in entries if r[0] == 'settlement']
    assert len(settlements) == 1 and settlements[0][1] == 2000
    assert settlements[0][2]['call_succeeded'] is True
    assert settlements[0][2]['provider_cost']['request_id'] == NATIVE


def test_initial_key_404_never_persists_trace_and_later_exact_bill_recovers(database, tmp_path, monkeypatch):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    trace = 'iad1::trace-1791330233666-aeb0e7021ee6'
    transport = RecoveryTransport(lookup_status=404,
        lost_error=br.ProviderTransportError('ReadTimeout', observed_status=200, deepline_job_id=trace))
    h, lease, token, connect, broker = setup(database, tmp_path, '16', transport)
    result = h.service.handle_provider(lease['run_id'], token, frame())
    assert result['status'] == 502 and result['call']['outcome'] == 'uncertain'
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT entry_doc FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'",
                    (result['call']['call_identity'],))
        doc = cur.fetchone()[0]['call']
    assert 'deepline_job_id' not in doc and trace not in json.dumps(doc)
    candidate = h.service.store.list_deepline_cost_reconciliations(h.round_id, run_id=lease['run_id'])[0]
    assert candidate['request_id'].startswith('ctx-tool-') and candidate['execution_key'].startswith('arena:')
    transport.lookup_status = 200
    settled = broker.reconcile_deepline_cost(candidate)
    assert settled['status'] == 'settled' and settled['actual_microusd'] == 2000
    assert broker.reconcile_deepline_cost(candidate)['status'] == 'settled'
    assert sum(r['method'] == 'POST' for r in transport.requests) == 1
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT terminal_response FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'",
                    (result['call']['call_identity'],))
        terminal = cur.fetchone()[0]
    # Billing recovery alone cannot claim the lost research was delivered.
    assert terminal['call_succeeded'] is False
    assert terminal['provider_cost']['request_id'] == NATIVE
