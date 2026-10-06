"""Pending async receipt, owned polling and exact settlement against the real ledger."""
import json
from dataclasses import replace

from lab_arena import broker as br
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import database, fixture_configuration, run
from tests.lab_arena.deepline_pending_async_receipt_test import AsyncTransport, JOB, NATIVE, POLL, execute, snapshot
from tests.lab_arena.reservation_response_loss_postgres_test import _context
from tests.lab_arena.test_lab_arena_broker import HOST_KEYS, price_table


def test_async_start_poll_and_delayed_bill_settle_once_without_hold(database, tmp_path, monkeypatch):
    monkeypatch.setattr(br, '_DEEPLINE_BILLING_MAX_ATTEMPTS', 1)
    h, lease, token, connect = run(database, tmp_path, '19', catalog=False)
    frozen = snapshot()
    with fixture_configuration(connect) as cur:
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc = "
            "jsonb_set(jsonb_set(configuration_doc,'{call_quotas,deepline}','0'),'{deepline_catalog}',%s::jsonb) "
            "WHERE round_id=%s", (json.dumps(frozen), h.round_id))
    context = replace(_context(lease, token, h.round_id), deepline_catalog=frozen)
    transport = AsyncTransport()
    broker = br.Broker(store=h.service.store, key_for=lambda provider: HOST_KEYS[provider],
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        funding_source_for=lambda _context: h.service.store.provider_funding(lease['run_id'], 'deepline')['funding_source'],
        price_table=price_table(), transport=transport, clock=h.clock)
    started = execute(broker, context)
    assert started.status == 200 and started.call['outcome'] == 'uncertain'
    before = len(transport.requests)
    assert execute(broker, context, POLL, 1, {'id': 'foreign-job'}).status >= 400
    assert execute(broker, replace(context, run_id='different-run'), POLL, 1, {'id': JOB}).status >= 400
    assert len(transport.requests) == before
    polled = execute(broker, context, POLL, 2, {'id': JOB})
    assert polled.status == 200 and polled.call['actual_microusd'] == 0
    candidate = h.service.store.list_deepline_cost_reconciliations(h.round_id, run_id=lease['run_id'])[0]
    assert candidate['request_id'] == NATIVE and candidate['execution_key'] is None
    transport.final = True
    result = broker.reconcile_deepline_cost(candidate)
    assert result['status'] == 'settled' and result['actual_microusd'] == 2000
    assert broker.reconcile_deepline_cost(candidate)['status'] == 'settled'
    requests_before_replay = len(transport.requests)
    replayed = execute(broker, context)
    assert replayed.call['idempotent'] is True and len(transport.requests) == requests_before_replay
    with connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT entry_kind,amount_microusd,entry_doc,terminal_response FROM public.lab_arena_ledger "
                    "WHERE call_identity=%s ORDER BY entry_id", (started.call['call_identity'],))
        entries = cur.fetchall()
    assert entries[0][0:2] == ('reservation', 0)
    uncertain = next(e for e in entries if e[0] == 'uncertain')
    assert uncertain[1] == 0 and uncertain[2]['call']['deepline_async_job_ids'] == [JOB]
    settlements = [e for e in entries if e[0] == 'settlement']
    assert len(settlements) == 1 and settlements[0][1] == 2000
    assert settlements[0][3]['call_succeeded'] is True
