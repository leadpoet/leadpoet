"""Temporary store diagnostics survive stage/claim telemetry without behavior changes."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability.otel_bootstrap import configure_arena_otel, STORE_ERROR_SPAN_FIELDS
from lab_arena import telemetry
from lab_arena.api import create_app
from lab_arena.store import (
    LOCK_TIMEOUT_RETRY_RPCS, ArenaStore, ArenaStoreError, ArenaStoreUnavailable,
    PostgrestTransport,
)
from tests.lab_arena.advance_round_telemetry_test import observed_service
from tests.lab_arena.test_lab_arena_api import StubService

CANARY = 'secret-prompt-SQL-https://private.invalid/token'
CLAIM = 'lab_arena_claim_assignment'


def _metadata(span):
    return {key.removeprefix('arena.store_'): value for key, value in span.attributes.items()
            if key in STORE_ERROR_SPAN_FIELDS}


def _native_store(kind, resource):
    requests, errors = [], []
    retried = kind == '40P01' or (kind == '55P03'
                                  and resource in LOCK_TIMEOUT_RETRY_RPCS)
    attempts = 4 if retried else 1
    def handle(request):
        requests.append(request)
        assert request.url.path == '/rest/v1/rpc/' + resource
        if len(requests) <= attempts:
            if kind == 'transport':
                error = httpx.ReadTimeout(CANARY)
                errors.append(error)
                raise error
            return httpx.Response(500, json={'code': kind, 'message': CANARY, 'details': CANARY})
        return httpx.Response(200, json={'status': 'no_pending'})
    client = httpx.Client(transport=httpx.MockTransport(handle))
    store = ArenaStore(PostgrestTransport('https://db.example', service_key='sb_secret_test', http_client=client))
    expected = dict(operation='rpc', resource=resource)
    expected.update(transport_error='ReadTimeout') if kind == 'transport' else expected.update(http_status=500, code=kind)
    return store, client, requests, errors, expected, attempts


@pytest.mark.parametrize('kind', ['transport', '55P03', '40P01'])
def test_actual_advance_stage_keeps_native_failure_and_existing_retry_policy(observed_service, kind):
    service, exporter, _ = observed_service
    store, client, requests, native, expected, attempts = _native_store(kind, 'lab_arena_expire_leases')
    raised = []
    def expire(round_id):
        try:
            return store.expire_leases(round_id)
        except ArenaStoreUnavailable as exc:
            raised.append(exc)
            raise
    service._store.expire_leases = expire
    try:
        with pytest.raises(ArenaStoreUnavailable) as caught:
            service.advance_round('private-round')
        assert caught.value is raised[0]
        assert caught.value.transaction_rolled_back is (kind != 'transport')
        assert caught.value.__cause__ is (native[0] if native else None)
        assert len(requests) == attempts
        failed, = [s for s in exporter.get_finished_spans() if s.attributes['arena.outcome'] == 'failed']
        assert failed.name == 'arena.advance_expire_leases'
        assert _metadata(failed) == expected
        assert CANARY not in str(failed)
        assert service._lock.acquire(blocking=False)
        service._lock.release()
    finally:
        client.close()


@pytest.mark.parametrize('kind', ['transport', '55P03', '40P01'])
def test_real_claim_route_exports_private_diagnostic_and_keeps_response_and_retries(monkeypatch, kind):
    store, database, requests, _, expected, attempts = _native_store(kind, CLAIM)
    class Service(StubService):
        def handle_claim(self, envelope):
            return store._transport.rpc(CLAIM, {'p_round_id': CANARY})
    exporter = InMemorySpanExporter()
    app = create_app(Service())
    recorder = configure_arena_otel(app, span_exporter=exporter)
    monkeypatch.setattr(telemetry, '_recorder', recorder)
    try:
        with TestClient(app) as api:
            failed = api.post('/arena/v1/runs/claim', json={'secret': CANARY})
            assert len(requests) == attempts
            healthy = api.post('/arena/v1/runs/claim', json={})
        assert failed.status_code == 503
        assert failed.json() == {'status': 'unavailable', 'code': 'arena_store_unavailable'}
        assert failed.headers['retry-after'] == '1'
        assert failed.headers['cache-control'] == 'no-store'
        assert failed.headers['x-content-type-options'] == 'nosniff'
        assert healthy.status_code == 200 and healthy.json() == {'status': 'no_pending'}
        assert len(requests) == attempts + 1
        failure_span, success_span = exporter.get_finished_spans()
        assert failure_span.name == 'POST /arena/v1/runs/claim'
        assert _metadata(failure_span) == expected
        assert failure_span.attributes['arena.denial'] == 'arena_store_unavailable'
        assert _metadata(success_span) == {}
        assert CANARY not in failed.text + str(failure_span) + str(success_span)
    finally:
        database.close()


@pytest.mark.parametrize('error_kind', [
    'unrelated', 'subclass', 'unknown_metadata',
    'base_subclass', 'base_unknown_metadata',
])
def test_stage_does_not_trust_spoofed_or_unknown_error_metadata(monkeypatch, error_kind):
    class Untrusted(ArenaStoreUnavailable):
        pass
    class UntrustedBase(ArenaStoreError):
        pass
    error = (ValueError(CANARY) if error_kind == 'unrelated' else
             Untrusted(CANARY) if error_kind == 'subclass' else
             UntrustedBase(CANARY) if error_kind == 'base_subclass' else
             ArenaStoreError(CANARY) if error_kind == 'base_unknown_metadata' else
             ArenaStoreUnavailable(CANARY))
    error.store_error = dict(operation='rpc', resource=CLAIM, code='55P03')
    if error_kind in ('unknown_metadata', 'base_unknown_metadata'):
        error.store_error = dict(operation=CANARY, resource=CANARY, message=CANARY)
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(telemetry, '_recorder', configure_arena_otel(None, span_exporter=exporter))
    with pytest.raises(type(error)) as caught:
        with telemetry.stage('advance_expire_leases'):
            raise error
    assert caught.value is error
    span, = exporter.get_finished_spans()
    assert _metadata(span) == {}
    assert CANARY not in str(span)


def test_metadata_projection_failure_cannot_replace_original_stage_error(monkeypatch):
    import lab_arena.store as store_module
    error = ArenaStoreUnavailable(CANARY, operation='rpc', resource=CLAIM, code='55P03')
    monkeypatch.setattr(store_module, 'safe_store_error_metadata', lambda *_: (_ for _ in ()).throw(RuntimeError(CANARY)))
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(telemetry, '_recorder', configure_arena_otel(None, span_exporter=exporter))
    with pytest.raises(ArenaStoreUnavailable) as caught:
        with telemetry.stage('advance_expire_leases'):
            raise error
    assert caught.value is error
    span, = exporter.get_finished_spans()
    assert _metadata(span) == {}


@pytest.mark.parametrize('error_kind', ['subclass', 'unknown_metadata', 'projection_failure', 'recorder_failure'])
def test_claim_response_survives_untrusted_or_failed_diagnostics(monkeypatch, error_kind):
    import lab_arena.store as store_module
    class Untrusted(ArenaStoreUnavailable):
        pass
    error = (Untrusted if error_kind == 'subclass' else ArenaStoreUnavailable)(CANARY)
    error.store_error = dict(operation='rpc', resource=CLAIM, code='55P03')
    if error_kind == 'unknown_metadata':
        error.store_error = {'operation': CANARY, 'resource': CANARY, 'message': CANARY}
    class Service(StubService):
        def handle_claim(self, envelope):
            raise error
    exporter = InMemorySpanExporter()
    app = create_app(Service())
    recorder = configure_arena_otel(app, span_exporter=exporter)
    monkeypatch.setattr(telemetry, '_recorder', recorder)
    if error_kind == 'projection_failure':
        monkeypatch.setattr(store_module, 'safe_store_error_metadata', lambda *_: (_ for _ in ()).throw(RuntimeError(CANARY)))
    elif error_kind == 'recorder_failure':
        monkeypatch.setattr(recorder, 'note_denial', lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError(CANARY)))
    with TestClient(app) as api:
        response = api.post('/arena/v1/runs/claim', json={})
    assert response.status_code == 503
    assert response.json() == {'status': 'unavailable', 'code': 'arena_store_unavailable'}
    assert response.headers['retry-after'] == '1'
    span, = exporter.get_finished_spans()
    assert _metadata(span) == {}
    assert CANARY not in response.text + str(span)


def test_claim_context_does_not_cross_concurrent_threadpool_requests(monkeypatch):
    barrier = Barrier(2)
    class Service(StubService):
        def handle_claim(self, envelope):
            barrier.wait(timeout=5)
            if envelope['fail']:
                raise ArenaStoreUnavailable(CANARY, operation='rpc', resource=CLAIM, code='55P03')
            return {'status': 'no_pending'}
    exporter = InMemorySpanExporter()
    app = create_app(Service())
    monkeypatch.setattr(telemetry, '_recorder', configure_arena_otel(app, span_exporter=exporter))
    with TestClient(app) as api, ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda fail: api.post('/arena/v1/runs/claim', json={'fail': fail}), [True, False]))
    assert [r.status_code for r in responses] == [503, 200]
    spans = {s.attributes['http.response.status_code']: s for s in exporter.get_finished_spans()}
    assert _metadata(spans[503]) == dict(operation='rpc', resource=CLAIM, code='55P03')
    assert _metadata(spans[200]) == {}


@pytest.mark.parametrize('patch', [
    {'arena.store_code': CANARY}, {'arena.store_resource': CANARY},
    {'arena.store_http_status': True}, {'arena.store_message': CANARY},
    {'arena.outcome': 'ok'}, {'arena.error_type': 'ValueError'},
])
@pytest.mark.parametrize('error_type', ['ArenaStoreUnavailable', 'ArenaStoreError'])
def test_final_stage_export_rejects_injected_diagnostics(monkeypatch, patch, error_type):
    exporter = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exporter)
    recorder.record('advance_expire_leases', 'failed', error_type=error_type,
                    store_error=dict(operation='rpc', resource=CLAIM, code='55P03'))
    valid, = exporter.get_finished_spans()
    exporter.clear()
    span = recorder._start(recorder._tracer, valid.name, None)
    span.set_attributes({**valid.attributes, **patch})
    span.end()
    assert exporter.get_finished_spans() == ()


@pytest.mark.parametrize('patch', [
    {'code': CANARY}, {'resource': CANARY}, {'operation': CANARY},
    {'http_status': True}, {'transport_error': CANARY}, {'message': CANARY},
])
def test_final_http_export_rejects_injected_diagnostics_without_changing_response(monkeypatch, patch):
    class Service(StubService):
        def handle_claim(self, envelope):
            raise ArenaStoreUnavailable(CANARY, operation='rpc', resource=CLAIM, code='55P03')
    exporter = InMemorySpanExporter()
    app = create_app(Service())
    recorder = configure_arena_otel(app, span_exporter=exporter)
    monkeypatch.setattr(telemetry, '_recorder', recorder)
    monkeypatch.setattr(recorder, 'take_store_error', lambda: {
        'operation': 'rpc', 'resource': CLAIM, 'code': '55P03', **patch,
    })
    with TestClient(app) as api:
        response = api.post('/arena/v1/runs/claim', json={})
    assert response.status_code == 503
    assert response.json() == {'status': 'unavailable', 'code': 'arena_store_unavailable'}
    assert exporter.get_finished_spans() == ()
