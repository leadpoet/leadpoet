"""Native operation receipts pass strict SQL recovery without rewriting history."""
import copy

import pytest

from lab_arena import broker as br
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.deepline_late_response_recovery_postgres_test import base_database, database, setup
from tests.lab_arena.deepline_native_cost_operation_test import CASES, NATIVE, NativeTransport


@pytest.mark.parametrize('operation,parameters,tool,provider', CASES[:2])
def test_legacy_exa_alias_recovers_with_real_postgres_guard(database, tmp_path, monkeypatch,
                                                          operation, parameters, tool, provider):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    transport = NativeTransport(tool, provider)
    label = '27' if operation == 'exa.search' else '28'
    h, lease, token, connect, broker = setup(database, tmp_path, label, transport)
    context = br.RunContext(run_id=lease['run_id'], assignment_id=lease['assignment_id'],
        icp_position=lease['icp_position'], lease_token_hash=hash_lease_token(token),
        miner_hotkey=lease['miner_hotkey'], submission_id=lease['submission_id'],
        stage=lease['stage'], kind='execute', attempt=lease['attempt'], round_id=h.round_id)
    request = dict(operation_id=operation, parameters=parameters, action_sequence=32, timeout_ms=60_000)
    first = broker.execute(context, **request)
    assert first.status == 502 and first.call['actual_microusd'] == 10000
    identity = first.call['call_identity']
    with connect() as connection, connection.cursor() as cur:
        cur.execute('SELECT entry_id,entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger '
                    'WHERE call_identity=%s ORDER BY entry_id', (identity,))
        before = cur.fetchall()
    assert before[-1][3]['provider_cost']['operation'] == tool
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT entry_doc FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='reservation'", (identity,))
        reservation = cur.fetchone()[0]
    arguments = dict(run_id=lease['run_id'], lease_token_hash=hash_lease_token(token),
        call_identity=identity, request_hash=reservation['request_hash'],
        execution_key=reservation['deepline_execution_key'],
        credential_fingerprint=reservation['credential_fingerprint'], request_id=NATIVE,
        operation=tool, actual_microusd=10000, lease_ttl_seconds=420,
        terminal_response=br._terminal_response_document(200, {'content-type': 'application/json'},
            b'{"results":[]}', call_succeeded=True, provider_cost={
                'operation': tool, 'request_id': NATIVE, 'basis': 'deepline_billing_credits_charged_x_0.10_usd',
                'units': '0.1', 'unit_name': 'credits'}))
    for change in ('cost_operation', 'amount', 'tool', 'owner', 'key'):
        invalid = copy.deepcopy(arguments)
        if change == 'cost_operation':
            invalid['terminal_response']['provider_cost']['operation'] = operation
        elif change == 'amount':
            invalid['actual_microusd'] = 20000
        elif change == 'tool':
            invalid['operation'] = 'other_tool'
        elif change == 'owner':
            invalid['run_id'] = 'foreign-run'
        else:
            invalid['execution_key'] = 'arena:' + 'f' * 64
        try:
            assert h.service.store.recover_deepline_response(**invalid)['status'] in ('conflict', 'stale')
        except ArenaStoreError:
            pass  # Malformed key input is rejected before acquiring a lease.
    with connect() as connection, connection.cursor() as cur:
        cur.execute('SELECT COUNT(*) FROM public.lab_arena_deepline_call_responses WHERE call_identity=%s', (identity,))
        assert cur.fetchone()[0] == 0
    transport.complete = True
    before_requests = len(transport.requests)
    recovered = broker.execute(context, **request)
    assert recovered.status == 200 and recovered.call['actual_microusd'] == 10000
    assert recovered.call['call_identity'] == identity
    assert all(r['method'] == 'GET' for r in transport.requests[before_requests:])
    assert sum(r['method'] == 'POST' for r in transport.requests) == 1
    with connect() as connection, connection.cursor() as cur:
        cur.execute('SELECT entry_id,entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger '
                    'WHERE call_identity=%s ORDER BY entry_id', (identity,))
        assert cur.fetchall() == before
        cur.execute('SELECT terminal_response FROM public.lab_arena_deepline_call_responses WHERE call_identity=%s', (identity,))
        saved = cur.fetchone()[0]
        assert saved['call_succeeded'] is True and saved['status'] == 200
        assert saved['provider_cost']['operation'] == tool
    count = len(transport.requests)
    assert broker.execute(context, **request).body == recovered.body
    assert len(transport.requests) == count
