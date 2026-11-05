"""No paid traffic: exercise the real broker with exact ledger history."""
from copy import deepcopy
from dataclasses import replace
import base64
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from lab_arena import broker as br
from tests.lab_arena.openrouter_failed_cost_recovery_test import HistoryStore
from tests.lab_arena.test_lab_arena_broker import CHAT, CONTEXT, HOST_KEYS, FakeTransport, make_broker

SUCCESS = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'usable answer'}}]}


class ReleaseStore(HistoryStore):
    def release_undispatched_call(self, *, run_id, lease_token_hash, call_identity, call_doc, reason, dispatch_entry_id=None):
        with self.lock:
            if self.stale:
                return {'status': 'stale'}
            call = self.calls[call_identity]
            if call['run_id'] != run_id or call['call_doc'] != call_doc:
                return {'status': 'reservation_mismatch'}
            if not (call['kind'] == 'reservation' and dispatch_entry_id is None
                    or call['kind'] == 'dispatch' and dispatch_entry_id == 2):
                return self._view(call)
            self.openrouter_capacity += call['amount']
            call.update(kind='recovery', reason=reason)
            return {'status': 'recovered', 'idempotent': False}


def execute(broker, *, context=CONTEXT, parameters=CHAT, cancel_requested=None):
    return broker.execute(context, operation_id='openrouter.chat', parameters=parameters,
        action_sequence=0, timeout_ms=30000, cancel_requested=cancel_requested)


@pytest.mark.parametrize('when', ['reservation', 'dispatch'])
@pytest.mark.parametrize('cause', ['cancel', 'deadline'])
def test_late_cancel_or_deadline_never_posts(monkeypatch, when, cause):
    now, cancelled = [0.0], [False]
    monkeypatch.setattr(br.time, 'monotonic', lambda: now[0])
    store = ReleaseStore()
    original = getattr(store, 'reserve_call' if when == 'reservation' else 'mark_dispatched')
    def delayed(**kwargs):
        result = original(**kwargs)
        if cause == 'cancel':
            cancelled[0] = True
        else:
            now[0] = 1000
        return result
    setattr(store, 'reserve_call' if when == 'reservation' else 'mark_dispatched', delayed)
    broker, _, transport = make_broker(store=store, transport=FakeTransport([(200, SUCCESS)]),
        openrouter_shared_gate=br.OpenRouterSharedGate(max_concurrency=1))
    result = execute(broker, cancel_requested=lambda: cancelled[0])
    assert transport.sent == []
    call = next(iter(store.calls.values()))
    assert call['kind'] == 'recovery'
    assert not any(name == 'settle' for name in store.log)
    assert result.call['outcome'] == 'not_dispatched'
    now[0], cancelled[0] = 0, False
    repeat = execute(broker)
    assert repeat.status != 200 and transport.sent == []


@pytest.mark.parametrize('run_kind,funding', [('execute', 'host'), ('execute', 'miner_key'), ('score', 'host')])
@pytest.mark.parametrize('failure', ['missing_cost', 'settlement'])
def test_success_is_durable_replayable_and_not_a_final_bill(run_kind, funding, failure):
    store = HistoryStore()
    payload = deepcopy(SUCCESS)
    if failure == 'settlement':
        payload['usage'] = {'cost': '0.001'}
        def rejected(**kwargs):
            raise br.ArenaStoreError('settlement contract mismatch')
        store.settle_call = rejected
    broker, _, transport = make_broker(store=store, transport=FakeTransport([(200, payload)]),
        funding_source_for=lambda _: funding, judge_models=(CHAT['model'],),
        credential_for=lambda _context, provider: HOST_KEYS[provider])
    context = replace(CONTEXT, kind=run_kind)
    first, repeat = execute(broker, context=context), execute(broker, context=context)
    assert first.status == repeat.status == 200
    assert (first.headers, first.body) == (repeat.headers, repeat.body)
    assert first.call['outcome'] == repeat.call['outcome'] == 'uncertain'
    assert first.call.get('actual_microusd') is repeat.call.get('actual_microusd') is None
    saved = next(iter(store.calls.values()))['uncertain_doc']
    assert saved['openrouter_terminal_response']['call_succeeded'] is True
    assert saved.get('known_actual_microusd') == (1000 if failure == 'settlement' else None)
    assert len(transport.sent) == 1


@pytest.mark.parametrize('change', ['request', 'credential', 'reservation', 'generation', 'secret', 'terminal_status'])
def test_success_replay_fails_closed_for_changed_or_conflicting_identity(change):
    broker, store, transport = make_broker(store=HistoryStore(), transport=FakeTransport([(200, SUCCESS)]))
    first = execute(broker)
    saved = next(iter(store.calls.values()))
    parameters = CHAT
    if change == 'request':
        parameters = {**CHAT, 'messages': [{'role': 'user', 'content': 'changed'}]}
        # A broken/conflicting identity lookup must never return the old reply.
        store.reserve_call = lambda **_: store._view(saved)
    elif change == 'credential':
        saved['uncertain_doc']['credential_fingerprint'] = 'sha256:' + 'a' * 64
    elif change == 'reservation':
        saved['call_doc']['request_hash'] = 'sha256:' + 'a' * 64
    elif change == 'generation':
        saved['uncertain_doc']['openrouter_generation_id'] = 'invalid/id'
    elif change == 'secret':
        saved['uncertain_doc']['openrouter_terminal_response']['body_b64'] = base64.b64encode(HOST_KEYS['openrouter'].encode()).decode()
    else:
        saved['uncertain_doc']['openrouter_terminal_response']['status'] = 404
    result = execute(broker, parameters=parameters)
    assert first.status == 200 and result.status != 200
    assert len(transport.sent) == 1


def test_private_receipt_alone_never_grants_success_replay():
    broker, store, transport = make_broker(store=HistoryStore(), transport=FakeTransport([(200, SUCCESS)]))
    execute(broker)
    saved = next(iter(store.calls.values()))['uncertain_doc']
    assert 'upstream_response_receipt' in saved
    saved.pop('openrouter_terminal_response')
    assert execute(broker).call['error_code'] == 'call_uncertain'
    assert len(transport.sent) == 1


def test_success_waits_for_retry_of_first_uncertainty_write(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(br.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(br.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    store = HistoryStore()
    write, attempts = store.mark_uncertain, []
    def interrupted(**kwargs):
        attempts.append(deepcopy(kwargs))
        if len(attempts) == 1:
            raise br.ArenaStoreUnavailable('unavailable')
        return write(**kwargs)
    store.mark_uncertain = interrupted
    broker, _, transport = make_broker(store=store, transport=FakeTransport([(200, SUCCESS)]))
    first, repeat = execute(broker), execute(broker)
    assert first.status == repeat.status == 200
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert len(transport.sent) == 1


@pytest.mark.parametrize('at', ['settlement', 'uncertainty'])
def test_stale_accounting_never_delivers_success(at):
    store = HistoryStore()
    payload = deepcopy(SUCCESS)
    if at == 'settlement':
        payload['usage'] = {'cost': '0.001'}
    setattr(store, 'settle_call' if at == 'settlement' else 'mark_uncertain', lambda **_: {'status': 'stale'})
    broker, _, transport = make_broker(store=store, transport=FakeTransport([(200, payload)]))
    assert execute(broker).call['error_code'] == 'lease_stale'
    assert len(transport.sent) == 1


def test_concurrent_exact_success_has_only_one_provider_post():
    entered, release = threading.Event(), threading.Event()
    class BlockingTransport(FakeTransport):
        def send(self, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().send(**kwargs)
    broker, _, transport = make_broker(store=HistoryStore(), transport=BlockingTransport([(200, SUCCESS)]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(execute, broker)
        assert entered.wait(5)
        second = pool.submit(execute, broker).result(timeout=5)
        assert second.call['error_code'] == 'call_uncertain'
        release.set()
        assert first.result(timeout=5).status == 200
    assert execute(broker).status == 200
    assert len(transport.sent) == 1


def test_committed_settlement_with_all_replies_lost_delivers_exact_success():
    store = HistoryStore()
    settle = store.settle_call
    def lost(**kwargs):
        settle(**kwargs)
        raise br.ArenaStoreError('settlement reply lost')
    store.settle_call = lost
    broker, _, transport = make_broker(store=store,
        transport=FakeTransport([(200, {**SUCCESS, 'usage': {'cost': '0.001'}})]))
    first, repeat = execute(broker), execute(broker)
    assert first.status == repeat.status == 200
    assert first.body == repeat.body
    assert first.call['outcome'] == repeat.call['outcome'] == 'settled'
    assert first.call['actual_microusd'] == repeat.call['actual_microusd'] == 1000
    assert len(transport.sent) == 1


def test_committed_uncertainty_survives_process_interruption():
    store = HistoryStore()
    persist = store.mark_uncertain
    def interrupted(**kwargs):
        persist(**kwargs)
        raise KeyboardInterrupt('process interrupted after durable write')
    store.mark_uncertain = interrupted
    transport = FakeTransport([(200, SUCCESS)])
    broker, _, _ = make_broker(store=store, transport=transport)
    with pytest.raises(KeyboardInterrupt):
        execute(broker)
    # No state is read from the interrupted broker instance.
    resumed, _, _ = make_broker(store=store, transport=transport)
    assert execute(resumed).status == 200
    assert len(transport.sent) == 1
