"""Observation storage and verification must leave shared control workers available."""
import asyncio
import threading

import anyio
import httpx
import pytest

from lab_arena.api import TRAINING_OBSERVATION_WORKER_LIMIT, create_app
from lab_arena.service import ServiceError
from tests.lab_arena.api_training_input_capacity_test import CaptureService
from tests.lab_arena.api_provider_capacity_test import HEADERS, post_provider, wait_until

DIGEST = 'sha256:' + '1' * 64
LIMIT = TRAINING_OBSERVATION_WORKER_LIMIT


class ObservationService(CaptureService):
    def __init__(self, default):
        super().__init__(default)
        self.observation_release = threading.Event()
        self.observation_calls = []
        self.observation_active = self.observation_maximum = 0
        self.capture_release.set()

    def _training_capture_run(self, *args, **kwargs):
        return {}

    _training_final_observation_capture_run = _training_capture_run

    def blocked(self, run_id, token, document):
        with self.lock:
            self.observation_calls.append(run_id)
            self.observation_active += 1
            self.observation_maximum = max(self.observation_maximum, self.observation_active)
        try:
            assert self.observation_release.wait(5), 'test did not release capture'
            if run_id == 'fail':
                raise ServiceError('training_observations_unavailable', 503)
            if run_id == 'unexpected':
                raise RuntimeError('synthetic failure')
            return {'status': 'stored'}
        finally:
            with self.lock:
                self.observation_active -= 1

    handle_training_observations = blocked
    handle_training_observation_segment = blocked

    def verify_training_observations(self, run_id, token, digest):
        self.blocked(run_id, token, digest)
        return digest


def request(client, kind, run_id):
    suffix = 'training-observation-segments' if kind == 'segment' else 'training-observations'
    path = '/arena/v1/runs/' + run_id + '/' + suffix
    if kind == 'verify':
        return asyncio.create_task(client.head(path, headers={**HEADERS, 'if-match': '"' + DIGEST + '"'}))
    return asyncio.create_task(client.post(path, json={}, headers=HEADERS))


def test_observation_burst_leaves_claim_completion_authorization_and_input_available():
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = ObservationService(default)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                providers = [post_provider(client, str(i)) for i in range(32)]
                captures = []
                try:
                    await wait_until(lambda: service.active == 32)
                    captures = [request(client, ('segment', 'final', 'verify')[i % 3], str(i)) for i in range(40)]
                    await wait_until(lambda: service.observation_active >= LIMIT)
                    controls = await asyncio.wait_for(asyncio.gather(
                        client.post('/arena/v1/runs/claim', json={}),
                        client.post('/arena/v1/runs/done/complete', json={'body': {'run_id': 'done'}}),
                        client.head('/arena/v1/runs/auth/training-observations', headers=HEADERS),
                        client.post('/arena/v1/runs/input/training-input', json={}, headers=HEADERS),
                    ), 1)
                    assert [r.status_code for r in controls] == [200, 200, 204, 200]
                    assert service.observation_maximum == LIMIT
                    assert default.borrowed_tokens == 32
                finally:
                    service.observation_release.set()
                    service.release.set()
                    responses = await asyncio.wait_for(asyncio.gather(*providers, *captures), 3)
                assert all(r.status_code in {200, 204} for r in responses)
                assert len(service.observation_calls) == 40
                assert service.observation_active == default.borrowed_tokens == 0
    asyncio.run(scenario())


@pytest.mark.parametrize('kind', ['segment', 'final', 'verify'])
@pytest.mark.parametrize('disconnect', [False, True])
def test_queued_cancel_or_disconnect_never_enters_service(kind, disconnect):
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = ObservationService(default)
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            captures = [request(client, kind, str(i)) for i in range(LIMIT)]
            await wait_until(lambda: service.observation_active >= LIMIT)
            messages = asyncio.Queue()
            messages.put_nowait({'type': 'http.request', 'body': b'{}', 'more_body': False})
            received_disconnect = asyncio.Event()
            sent = []
            async def receive():
                message = await messages.get()
                if message['type'] == 'http.disconnect':
                    received_disconnect.set()
                return message
            async def send(message):
                sent.append(message)
            suffix = 'training-observation-segments' if kind == 'segment' else 'training-observations'
            path = '/arena/v1/runs/queued/' + suffix
            headers = [(b'x-lab-arena-lease', HEADERS['x-lab-arena-lease'].encode()), (b'content-type', b'application/json')]
            if kind == 'verify':
                headers.append((b'if-match', ('"' + DIGEST + '"').encode()))
            scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                     'method': 'HEAD' if kind == 'verify' else 'POST', 'scheme': 'http',
                     'path': path, 'raw_path': path.encode(), 'query_string': b'', 'root_path': '',
                     'headers': headers, 'client': ('test', 1), 'server': ('test', 80)}
            queued = asyncio.create_task(app(scope, receive, send))
            try:
                await asyncio.sleep(.03)
                if disconnect:
                    messages.put_nowait({'type': 'http.disconnect'})
                    await asyncio.wait_for(received_disconnect.wait(), 1)
                    await asyncio.wait_for(queued, 1)
                    assert next(m['status'] for m in sent if m['type'] == 'http.response.start') == 499
                else:
                    queued.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await queued
                assert 'queued' not in service.observation_calls
            finally:
                service.observation_release.set()
                await asyncio.wait_for(asyncio.gather(*captures), 3)
            assert 'queued' not in service.observation_calls
            assert service.observation_active == default.borrowed_tokens == 0
            assert (await request(client, kind, 'after')).status_code in {200, 204}
    asyncio.run(scenario())


@pytest.mark.parametrize('kind', ['segment', 'final', 'verify'])
@pytest.mark.parametrize('run_id', ['fail', 'unexpected'])
def test_capture_failure_releases_permit_and_preserves_error(kind, run_id):
    async def scenario():
        service = ObservationService(anyio.to_thread.current_default_thread_limiter())
        service.observation_release.set()
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            if run_id == 'unexpected':
                with pytest.raises(RuntimeError, match='synthetic failure'):
                    await request(client, kind, run_id)
            else:
                response = await request(client, kind, run_id)
                assert response.status_code == 503
            responses = await asyncio.gather(*(request(client, kind, str(i)) for i in range(LIMIT + 1)))
            assert all(r.status_code in {200, 204} for r in responses)
            assert service.observation_active == 0
    asyncio.run(scenario())


@pytest.mark.parametrize('kind', ['segment', 'final', 'verify'])
@pytest.mark.parametrize('cancel_count', [1, 2])
def test_active_cancellation_retains_capture_permit_until_worker_finishes(kind, cancel_count):
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = ObservationService(default)
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            captures = [request(client, kind, '0')]
            try:
                # POST authorization uses separate workers, so task creation
                # order does not establish admission order. Prove this worker
                # is active before queuing peers and cancelling its request.
                await wait_until(lambda: service.observation_calls == ['0'])
                captures.extend(request(client, kind, str(i)) for i in range(1, LIMIT + 1))
                await wait_until(lambda: service.observation_active >= LIMIT)
                for _ in range(cancel_count):
                    captures[0].cancel()
                    await asyncio.sleep(.02)
                assert not captures[0].done()
                assert service.observation_active == len(service.observation_calls) == LIMIT
                assert default.borrowed_tokens == 0
            finally:
                service.observation_release.set()
                responses = await asyncio.wait_for(asyncio.gather(*captures, return_exceptions=True), 3)
            assert isinstance(responses[0], asyncio.CancelledError)
            assert all(r.status_code in {200, 204} for r in responses[1:])
            assert service.observation_maximum == LIMIT
            assert service.observation_active == default.borrowed_tokens == 0
    asyncio.run(scenario())


@pytest.mark.parametrize('kind', ['segment', 'final', 'verify'])
def test_http_disconnect_retains_active_observation_worker_and_permit(kind):
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = ObservationService(default)
        app = create_app(service)
        messages = asyncio.Queue()
        body_read = False
        disconnect_read = asyncio.Event()
        sent = []

        async def receive():
            nonlocal body_read
            if not body_read:
                body_read = True
                return {"type": "http.request", "body": b'{"input":"active"}', "more_body": False}
            message = await messages.get()
            disconnect_read.set()
            return message

        async def send(message):
            sent.append(message)

        suffix = 'training-observation-segments' if kind == 'segment' else 'training-observations'
        path = '/arena/v1/runs/active/' + suffix
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "HEAD" if kind == "verify" else "POST", "scheme": "http", "path": path,
            "raw_path": path.encode(), "query_string": b"",
            "headers": [(b"content-type", b"application/json"),
                        (b"x-lab-arena-lease", HEADERS['x-lab-arena-lease'].encode())],
            "client": ("127.0.0.1", 1234), "server": ("test", 80),
        }
        if kind == 'verify':
            scope['headers'].append((b'if-match', ('\"' + DIGEST + '\"').encode()))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            active = asyncio.create_task(app(scope, receive, send))
            await wait_until(lambda: service.observation_active == 1)
            others = [request(client, kind, str(i)) for i in range(LIMIT)]
            try:
                await wait_until(lambda: service.observation_active == LIMIT)
                messages.put_nowait({"type": "http.disconnect"})
                await asyncio.wait_for(disconnect_read.wait(), 1)
                assert not active.done()
                assert service.observation_active == len(service.observation_calls) == LIMIT
                assert default.borrowed_tokens == 0
            finally:
                service.observation_release.set()
                responses = await asyncio.wait_for(asyncio.gather(*others, active), 3)
            assert all(response.status_code in {200, 204} for response in responses[:-1])
            assert next(m["status"] for m in sent if m["type"] == "http.response.start") == 499
            assert service.observation_maximum == LIMIT
            assert len(service.observation_calls) == LIMIT + 1
            assert service.observation_active == default.borrowed_tokens == 0
    asyncio.run(scenario())
