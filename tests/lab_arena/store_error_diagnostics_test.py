"""Store diagnostics are structured, bounded, private, and behavior neutral."""
import json
from types import SimpleNamespace

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability.otel_bootstrap import configure_arena_otel, STORE_ERROR_SPAN_FIELDS
from lab_arena import telemetry
from lab_arena.store import (
    ArenaStore, ArenaStoreError, ArenaStoreUnavailable, PostgrestTransport,
    PsycopgTransport, safe_store_error_metadata, hash_lease_token,
)
from tests.lab_arena.test_lab_arena_broker import (
    CHAT, FakeLedgerStore, FakeTransport, make_broker,
)
from tests.lab_arena.trajectory_test import _run, _service_for_trajectory

CANARY = 'sk-or-v1-private-secret-message-prompt-SQL-https://private.invalid'
RPC = 'lab_arena_reserve_call'


@pytest.mark.parametrize('operation,resource,status,code', [
    ('rpc', RPC, 500, '42501'),
    ('select', 'lab_arena_runs', 503, 'PGRST003'),
])
def test_http_origin_records_only_machine_diagnostics(operation, resource, status, code):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={'code': code, 'message': CANARY, 'details': CANARY, 'hint': CANARY})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        store = PostgrestTransport('https://db.example', service_key='sb_secret_test', http_client=client)
        with pytest.raises(ArenaStoreError) as caught:
            store.rpc(resource, {'p_run_id': CANARY}) if operation == 'rpc' else store.select(resource)
    assert caught.value.store_error == dict(operation=operation, resource=resource, http_status=status, code=code)
    assert CANARY not in json.dumps(caught.value.store_error)
    assert len(requests) == (2 if operation == 'select' else 1)


@pytest.mark.parametrize('operation,resource,attempts', [
    ('rpc', RPC, 1), ('select', 'lab_arena_runs', 2),
])
def test_transport_origin_retains_existing_retry_and_ambiguity(operation, resource, attempts):
    requests = []
    def handle(request):
        requests.append(request)
        raise httpx.ReadTimeout(CANARY)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        store = PostgrestTransport('https://db.example', service_key='sb_secret_test', http_client=client)
        with pytest.raises(ArenaStoreUnavailable) as caught:
            store.rpc(resource, {}) if operation == 'rpc' else store.select(resource)
    assert caught.value.store_error == dict(operation=operation, resource=resource, transport_error='ReadTimeout')
    assert caught.value.transaction_rolled_back is False
    assert len(requests) == attempts
    assert CANARY not in json.dumps(caught.value.store_error)


@pytest.mark.parametrize('operation,resource', [('rpc', RPC), ('select', 'lab_arena_runs')])
def test_psycopg_origin_uses_sqlstate_not_detail(operation, resource):
    class DatabaseFailure(Exception):
        pgcode = '42501'
        diag = SimpleNamespace(message_detail=CANARY, context=CANARY)
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, *_args): raise DatabaseFailure(CANARY)
    connection = SimpleNamespace(cursor=Cursor, close=lambda: None)
    store = PsycopgTransport(lambda: connection, role=None)
    try:
        with pytest.raises(ArenaStoreError) as caught:
            store.rpc(resource, {}) if operation == 'rpc' else store.select(resource)
        assert caught.value.store_error == dict(operation=operation, resource=resource, code='42501')
        assert CANARY not in json.dumps(caught.value.store_error)
    finally:
        store.close()


@pytest.mark.parametrize('patch', [
    {'resource': CANARY}, {'operation': CANARY}, {'code': CANARY},
    {'code': {'message': CANARY}}, {'transport_error': CANARY},
    {'http_status': True}, {'http_status': 999}, {'message': CANARY, 'args': CANARY, 'url': CANARY},
])
def test_untrusted_fields_cannot_escape_projection(patch):
    value = dict(operation='rpc', resource=RPC)
    value.update(patch)
    projected = safe_store_error_metadata(value)
    assert CANARY not in json.dumps(projected)
    assert set(projected) <= {'operation', 'resource', 'http_status', 'code', 'transport_error'}
    assert 'http_status' not in projected


def _provider_service(monkeypatch, *, fail=True, sink_failure=False):
    ledger = FakeLedgerStore()
    requests, events = [], []
    failure = []
    def handle(request):
        requests.append(request)
        assert request.url.path == '/rest/v1/rpc/' + RPC
        if fail:
            return httpx.Response(500, json={'code': '42501', 'message': CANARY, 'details': CANARY})
        args = {key.removeprefix('p_'): value for key, value in json.loads(request.content).items()}
        return httpx.Response(200, json=ledger.reserve_call(**args))
    client = httpx.Client(transport=httpx.MockTransport(handle))
    durable = ArenaStore(PostgrestTransport('https://db.example', service_key='sb_secret_test', http_client=client))
    run = {**_run(), 'round_id': 'arena-2026-10-16', 'lease_token_hash': hash_lease_token('lease-token')}
    class Store:
        def get_run(self, _run_id): return run
        def reserve_call(self, **kwargs):
            try: return durable.reserve_call(**kwargs)
            except ArenaStoreError as exc:
                failure.append(exc)
                raise
        def append_trajectory_events(self, _run, _lease, items):
            if sink_failure: raise ArenaStoreError(CANARY)
            events.extend(items)
            return {'status': 'accepted', 'accepted': len(items)}
        def __getattr__(self, name): return getattr(ledger, name)
    provider = FakeTransport([(200, {'model': CHAT['model'], 'choices': [], 'usage': {'cost': '0.0000123'}})])
    broker, _, _ = make_broker(store=Store(), transport=provider, provider_funding_source_for=lambda *_: 'miner_key')
    service = _service_for_trajectory(Store(), broker)
    exporter = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exporter)
    monkeypatch.setattr(telemetry, '_recorder', recorder)
    frame = dict(operation_id='openrouter.chat', parameters=CHAT, timeout_ms=120000, action_sequence=31)
    return service, frame, ledger, provider, requests, events, failure, exporter, client


@pytest.mark.parametrize('sink_failure', [False, True])
def test_real_broker_store_failure_survives_service_and_private_export(monkeypatch, sink_failure):
    service, frame, ledger, provider, requests, events, failure, exporter, client = _provider_service(monkeypatch, sink_failure=sink_failure)
    try:
        with pytest.raises(ArenaStoreError) as caught:
            service.handle_provider('run-1', 'lease-token', frame)
        assert caught.value is failure[0]
        assert ledger.calls == {} and ledger.log == []
        assert provider.sent == [] and len(requests) == 1
        expected = dict(operation='rpc', resource=RPC, http_status=500, code='42501')
        if not sink_failure:
            error, = [e for e in events if e['kind'] == 'provider.error']
            assert error['content']['store_error'] == expected
            assert error['content']['action_sequence'] == 31
        span, = exporter.get_finished_spans()
        attrs = dict(span.attributes)
        assert {k.removeprefix('arena.store_'): v for k, v in attrs.items() if k in STORE_ERROR_SPAN_FIELDS} == expected
        assert attrs['arena.run_id'] == 'run-1'
        assert attrs['arena.error_type'] == 'ArenaStoreError'
        assert attrs['arena.http_status'] == 0 and attrs['arena.provider_status'] == 0
        # Request projections retain ordinary fixture prompts; the new error
        # surfaces export neither exception details nor request data.
        assert CANARY not in str(span) + str([e for e in events if e['kind']=='provider.error'])
    finally:
        client.close()


def test_success_has_same_dispatch_charge_and_no_store_diagnostics(monkeypatch):
    service, frame, ledger, provider, requests, events, failure, exporter, client = _provider_service(monkeypatch, fail=False)
    try:
        result = service.handle_provider('run-1', 'lease-token', frame)
        assert result['status'] == 200 and result['call']['actual_microusd'] == 13
        assert len(provider.sent) == 1 and len(requests) == 1 and failure == []
        assert len(ledger.calls) == 1 and next(iter(ledger.calls.values()))['actual'] == 13
        assert [e['kind'] for e in events] == ['provider.request', 'provider.response']
        span, = exporter.get_finished_spans()
        assert not (set(span.attributes) & STORE_ERROR_SPAN_FIELDS)
    finally:
        client.close()


def test_export_revalidates_metadata_and_drops_unknown_fields(monkeypatch):
    exporter = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exporter)
    recorder.record_provider('openrouter', 'openrouter.chat', 'failed', store_error={
        'operation': 'rpc', 'resource': RPC, 'code': CANARY, 'transport_error': CANARY,
        'http_status': True, 'message': CANARY, 'url': CANARY,
    })
    span, = exporter.get_finished_spans()
    assert {k: v for k, v in span.attributes.items() if k in STORE_ERROR_SPAN_FIELDS} == {
        'arena.store_operation': 'rpc', 'arena.store_resource': RPC,
    }
    assert CANARY not in str(span)


@pytest.mark.parametrize('patch', [
    {'arena.store_code': CANARY}, {'arena.store_resource': CANARY},
    {'arena.store_operation': CANARY}, {'arena.store_transport_error': CANARY},
    {'arena.store_http_status': True}, {'arena.store_message': CANARY},
    {'arena.store_resource': None},
])
def test_final_export_boundary_rejects_injected_store_attributes(patch):
    exporter = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exporter)
    recorder.record_provider('openrouter', 'openrouter.chat', 'failed', store_error={
        'operation': 'rpc', 'resource': RPC, 'http_status': 500, 'code': '42501',
    })
    valid, = exporter.get_finished_spans()
    attrs = {**valid.attributes, **patch}
    exporter.clear()
    span = recorder._start(recorder._provider_tracer, 'arena.provider.openrouter', None)
    span.set_attributes(attrs)
    span.end()
    assert exporter.get_finished_spans() == ()


def test_nonstore_exception_cannot_claim_store_diagnostics(monkeypatch):
    service, frame, ledger, provider, requests, events, failure, exporter, client = _provider_service(monkeypatch)
    original = ValueError(CANARY)
    original.store_error = {'operation': 'rpc', 'resource': RPC, 'code': '42501'}
    service._broker_for = lambda _: SimpleNamespace(execute=lambda *_a, **_k: (_ for _ in ()).throw(original))
    try:
        with pytest.raises(ValueError) as caught:
            service.handle_provider('run-1', 'lease-token', frame)
        assert caught.value is original
        assert requests == [] and provider.sent == [] and ledger.calls == {}
        error, = [e for e in events if e['kind']=='provider.error']
        assert 'store_error' not in error['content']
        span, = exporter.get_finished_spans()
        assert not (set(span.attributes) & STORE_ERROR_SPAN_FIELDS)
        assert CANARY not in str(span) + str(error)
    finally:
        client.close()
