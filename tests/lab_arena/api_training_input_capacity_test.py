"""Input capture retains independent bounded dispatch without changing stored evidence."""
from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import anyio
import httpx
import pytest

from lab_arena import training_input
from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import hash_lease_token
from tests.lab_arena.api_provider_capacity_test import (
    BlockingService, HEADERS, POOL, PROVIDER_WORKER_LIMIT, post_provider, wait_until,
)


CAPTURE_LIMIT = 2
TOKEN = HEADERS["x-lab-arena-lease"]
CLAIM_HASH = "sha256:" + "c" * 64
NOW = datetime(2026, 11, 2, tzinfo=timezone.utc)


def post_input(client, run_id, document=None, *, token=TOKEN):
    return asyncio.create_task(client.post(
        "/arena/v1/runs/" + run_id + "/training-input",
        json={"input": run_id} if document is None else document,
        headers={"x-lab-arena-lease": token},
    ))


class CaptureService(BlockingService):
    def __init__(self, default):
        super().__init__(default)
        self.capture_release = threading.Event()
        self.input_calls = []
        self.capture_active = 0
        self.capture_maximum = 0

    def handle_training_input(self, run_id, token, document):
        with self.lock:
            self.input_calls.append((run_id, token, document))
            self.capture_active += 1
            self.capture_maximum = max(self.capture_maximum, self.capture_active)
        try:
            if not self.capture_release.wait(5):
                raise AssertionError("test did not release input capture")
            if run_id == "fail":
                raise ServiceError("training_input_unavailable", 503)
            return {"status": "stored"}
        finally:
            with self.lock:
                self.capture_active -= 1


def test_training_input_queue_preserves_provider_and_control_workers():
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = CaptureService(default)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                providers = [post_provider(client, str(i)) for i in range(PROVIDER_WORKER_LIMIT)]
                captures = []
                try:
                    await wait_until(lambda: service.active == PROVIDER_WORKER_LIMIT)
                    captures = [post_input(client, "input-" + str(i)) for i in range(8)]
                    await wait_until(lambda: service.capture_active >= CAPTURE_LIMIT)
                    controls = await asyncio.wait_for(asyncio.gather(
                        client.post("/arena/v1/runs/claim", json={}),
                        client.post("/arena/v1/runs/done/complete", json={"body": {"run_id": "done"}}),
                    ), 1)
                    assert all(response.status_code == 200 for response in controls)
                    assert service.capture_active == len(service.input_calls) == CAPTURE_LIMIT
                    assert default.total_tokens == POOL == 64
                    assert default.borrowed_tokens == PROVIDER_WORKER_LIMIT
                    assert service.active == service.maximum == PROVIDER_WORKER_LIMIT == 32
                finally:
                    service.capture_release.set()
                    service.release.set()
                    responses = await asyncio.wait_for(asyncio.gather(*providers, *captures), 3)
                assert all(response.status_code == 200 for response in responses)
                assert len(service.input_calls) == len(captures)
                assert service.capture_maximum == CAPTURE_LIMIT
                assert all(token == TOKEN and document == {"input": run_id}
                           for run_id, token, document in service.input_calls)
                assert service.capture_active == service.active == default.borrowed_tokens == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("queue", ["capture", "default_saturated"])
def test_cancelled_queued_capture_never_enters_service_and_returns_permit(queue):
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = CaptureService(default)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                blockers = []
                cancelled = None
                try:
                    if queue == "default_saturated":
                        blockers = [asyncio.create_task(client.get("/arena/v1/competition")) for _ in range(POOL)]
                        await wait_until(lambda: service.other_started == POOL)
                    blockers.extend(post_input(client, str(i)) for i in range(CAPTURE_LIMIT))
                    await wait_until(lambda: service.capture_active == CAPTURE_LIMIT)
                    cancelled = post_input(client, "cancelled")
                    await anyio.sleep(0.02)
                    assert not cancelled.done()
                    cancelled.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await cancelled
                    assert all(row[0] != "cancelled" for row in service.input_calls)
                finally:
                    service.capture_release.set()
                    service.release.set()
                    await asyncio.wait_for(asyncio.gather(*blockers), 3)
                    if cancelled is not None and not cancelled.done():
                        cancelled.cancel()
                        await asyncio.gather(cancelled, return_exceptions=True)
                assert default.borrowed_tokens == 0
                service.capture_release.clear()
                next_wave = [post_input(client, "after-" + str(i)) for i in range(CAPTURE_LIMIT)]
                try:
                    await wait_until(lambda: service.capture_active == CAPTURE_LIMIT)
                finally:
                    service.capture_release.set()
                    responses = await asyncio.wait_for(asyncio.gather(*next_wave), 3)
                assert all(response.status_code == 200 for response in responses)
                assert service.capture_active == default.borrowed_tokens == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_count", [1, 2])
def test_cancelled_active_capture_retains_permit_until_its_worker_finishes(cancel_count):
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = CaptureService(default)
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            captures = [post_input(client, str(i)) for i in range(CAPTURE_LIMIT + 1)]
            try:
                await wait_until(lambda: service.capture_active == CAPTURE_LIMIT)
                for _ in range(cancel_count):
                    captures[0].cancel()
                    await anyio.sleep(0.02)
                assert not captures[0].done()
                assert service.capture_active == len(service.input_calls) == CAPTURE_LIMIT
                assert default.borrowed_tokens == 0
            finally:
                service.capture_release.set()
                responses = await asyncio.wait_for(asyncio.gather(*captures, return_exceptions=True), 3)
            assert isinstance(responses[0], asyncio.CancelledError)
            assert all(response.status_code == 200 for response in responses[1:])
            assert service.capture_maximum == CAPTURE_LIMIT
            assert service.capture_active == default.borrowed_tokens == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("queue", ["capture", "default_saturated"])
@pytest.mark.parametrize("large_body", [False, True])
def test_disconnected_queued_capture_never_enters_service(queue, large_body):
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = CaptureService(default)
        app = create_app(service)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                blockers = []
                if queue == "default_saturated":
                    blockers = [asyncio.create_task(client.get("/arena/v1/competition")) for _ in range(POOL)]
                    await wait_until(lambda: service.other_started == POOL)
                blockers.extend(post_input(client, str(i)) for i in range(CAPTURE_LIMIT))
                await wait_until(lambda: service.capture_active == CAPTURE_LIMIT)
                body_read = asyncio.Event()
                messages = asyncio.Queue()
                sent = []
                raw = json.dumps({"input": "x" * 2_204_160 if large_body else "disconnected"}).encode()

                async def receive():
                    if not body_read.is_set():
                        body_read.set()
                        return {"type": "http.request", "body": raw, "more_body": False}
                    return await messages.get()

                async def send(message):
                    sent.append(message)

                path = "/arena/v1/runs/disconnected/training-input"
                scope = {
                    "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                    "method": "POST", "scheme": "http", "path": path,
                    "raw_path": path.encode(), "query_string": b"",
                    "headers": [(b"content-type", b"application/json"),
                                (b"x-lab-arena-lease", TOKEN.encode())],
                    "client": ("127.0.0.1", 1234), "server": ("test", 80),
                }
                request = asyncio.create_task(app(scope, receive, send))
                try:
                    await body_read.wait()
                    await anyio.sleep(0.02)
                    assert not request.done()
                    messages.put_nowait({"type": "http.disconnect"})
                    # A listener must remove disconnected work while capacity
                    # is still held, rather than waiting to poll at dispatch.
                    await asyncio.wait_for(asyncio.shield(request), 0.5)
                finally:
                    service.capture_release.set()
                    service.release.set()
                    await asyncio.wait_for(asyncio.gather(*blockers, request), 3)
                assert all(row[0] != "disconnected" for row in service.input_calls)
                assert next(m["status"] for m in sent if m["type"] == "http.response.start") == 499
                assert service.capture_active == default.borrowed_tokens == 0
                service.capture_release.clear()
                next_wave = [post_input(client, "after-" + str(i)) for i in range(CAPTURE_LIMIT)]
                try:
                    await wait_until(lambda: service.capture_active == CAPTURE_LIMIT)
                finally:
                    service.capture_release.set()
                    assert all(r.status_code == 200 for r in await asyncio.gather(*next_wave))
                assert default.borrowed_tokens == 0
    asyncio.run(scenario())


def test_http_disconnect_keeps_active_worker_and_permit_until_finished():
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = CaptureService(default)
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

        path = "/arena/v1/runs/active/training-input"
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": path,
            "raw_path": path.encode(), "query_string": b"",
            "headers": [(b"content-type", b"application/json"),
                        (b"x-lab-arena-lease", TOKEN.encode())],
            "client": ("127.0.0.1", 1234), "server": ("test", 80),
        }
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            active = asyncio.create_task(app(scope, receive, send))
            await wait_until(lambda: service.capture_active == 1)
            others = [post_input(client, str(i)) for i in range(CAPTURE_LIMIT)]
            try:
                await wait_until(lambda: service.capture_active == CAPTURE_LIMIT)
                messages.put_nowait({"type": "http.disconnect"})
                await asyncio.wait_for(disconnect_read.wait(), 1)
                assert not active.done()
                assert service.capture_active == len(service.input_calls) == CAPTURE_LIMIT
                assert default.borrowed_tokens == 0
            finally:
                service.capture_release.set()
                responses = await asyncio.wait_for(asyncio.gather(*others, active), 3)
            assert all(response.status_code == 200 for response in responses[:-1])
            assert next(m["status"] for m in sent if m["type"] == "http.response.start") == 499
            assert service.capture_maximum == CAPTURE_LIMIT
            assert len(service.input_calls) == CAPTURE_LIMIT + 1
            assert service.capture_active == default.borrowed_tokens == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [ServiceError("training_input_unavailable", 503), RuntimeError("synthetic storage failure")])
def test_capture_exception_returns_input_permit(failure):
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = CaptureService(default)
        original = service.handle_training_input

        def fail(run_id, token, document):
            if run_id == "fail":
                raise failure
            return original(run_id, token, document)

        service.handle_training_input = fail
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            if isinstance(failure, ServiceError):
                response = await post_input(client, "fail")
                assert response.status_code == 503
                assert response.json() == {"status": "rejected", "code": "training_input_unavailable"}
                assert response.headers["cache-control"] == "no-store"
                assert response.headers["x-content-type-options"] == "nosniff"
            else:
                with pytest.raises(RuntimeError) as caught:
                    await post_input(client, "fail")
                assert caught.value is failure
            assert default.borrowed_tokens == 0
            captures = [post_input(client, str(i)) for i in range(CAPTURE_LIMIT)]
            try:
                await wait_until(lambda: service.capture_active == CAPTURE_LIMIT)
            finally:
                service.capture_release.set()
                responses = await asyncio.wait_for(asyncio.gather(*captures), 3)
            assert all(response.status_code == 200 for response in responses)
            assert service.capture_active == default.borrowed_tokens == 0
    asyncio.run(scenario())


def real_service(objects):
    def get_run(run_id):
        return {
            "run_id": run_id, "round_id": "arena-test", "kind": "score", "status": "leased",
            "lease_token_hash": hash_lease_token(TOKEN), "lease_generation": 1,
            "claim_request_hash": CLAIM_HASH, "lease_expires_at": (NOW + timedelta(hours=1)).isoformat(),
        }
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(get_run=get_run)
    service._config = SimpleNamespace(training_object_store=objects)
    service._clock = lambda: NOW
    service._require_round_ownership = lambda _: None
    return service


def test_slow_capture_leaves_second_lane_for_exact_private_inputs():
    async def scenario():
        slow_started = threading.Event()
        release_slow = threading.Event()
        saved = {}
        lock = threading.Lock()

        class Objects:
            def put(self, ref, data):
                if "/slow/" in ref:
                    slow_started.set()
                    assert release_slow.wait(5)
                with lock:
                    saved[ref] = data

        service = real_service(Objects())
        document = {"api_key": "synthetic-secret", "public": "complete input", "schema": {"maximum": 2 ** 64 - 1}}
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            slow = post_input(client, "slow", document)
            try:
                await wait_until(slow_started.is_set)
                healthy = await asyncio.wait_for(asyncio.gather(
                    *(post_input(client, "healthy-" + str(i), document) for i in range(5))
                ), 2)
                assert all(response.status_code == 200 for response in healthy)
                assert not slow.done()
            finally:
                release_slow.set()
                assert (await asyncio.wait_for(slow, 2)).status_code == 200
        for run_id in ["slow", *("healthy-" + str(i) for i in range(5))]:
            expected, compressed = training_input.build_snapshot(
                run_id, "score", document, lease_generation=1, claim_request_hash=CLAIM_HASH, secrets=(TOKEN,),
            )
            assert saved[training_input.object_ref(run_id, 1)] == compressed
            assert expected["privacy"] == {"redacted": True, "redactions": ["credential"]}
            assert expected["input"]["schema"]["maximum"] == 2 ** 64 - 1
        assert anyio.to_thread.current_default_thread_limiter().borrowed_tokens == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("control", ["bad_header", "wrong_lease", "not_object", "nonfinite", "control_character", "oversize"])
def test_input_auth_and_schema_controls_remain_unchanged(control):
    async def scenario():
        stored = []
        service = real_service(SimpleNamespace(put=lambda *args: stored.append(args)))
        app = create_app(service)
        raw, token, expected = b'{"public":"input"}', TOKEN, 200
        if control == "bad_header":
            token, expected = "invalid", 401
        elif control == "wrong_lease":
            token, expected = "b" * 64, 401
        elif control == "not_object":
            raw, expected = b"[]", 400
        elif control == "nonfinite":
            raw, expected = b'{"value":NaN}', 400
        elif control == "control_character":
            raw, expected = json.dumps({"value": "\x00"}).encode(), 400
        elif control == "oversize":
            raw, expected = json.dumps({"value": "x" * training_input.MAX_INPUT_BYTES}).encode(), 413
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/arena/v1/runs/r1/training-input", content=raw,
                                         headers={"x-lab-arena-lease": token, "content-type": "application/json"})
            assert response.status_code == expected
            assert stored == []
            assert (await post_input(client, "r1")).status_code == 200
            assert len(stored) == 1
            assert anyio.to_thread.current_default_thread_limiter().borrowed_tokens == 0
    asyncio.run(scenario())


def test_input_capture_progresses_when_other_routes_hold_default_workers():
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = CaptureService(default)
        service.capture_release.set()
        app = create_app(service)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                providers = [post_provider(client, str(i)) for i in range(PROVIDER_WORKER_LIMIT)]
                others = []
                capture = None
                try:
                    await wait_until(lambda: service.active == PROVIDER_WORKER_LIMIT)
                    others = [asyncio.create_task(client.get("/arena/v1/competition"))
                              for _ in range(POOL - PROVIDER_WORKER_LIMIT)]
                    await wait_until(lambda: service.other_started == POOL - PROVIDER_WORKER_LIMIT)
                    assert default.borrowed_tokens == POOL
                    capture = post_input(client, "independent-input")
                    done, _ = await asyncio.wait([capture], timeout=0.5)
                    assert capture in done, "input capture waits behind unrelated default-pool work"
                    response = capture.result()
                    assert response.status_code == 200 and response.json() == {"status": "stored"}
                    assert response.headers["cache-control"] == "no-store"
                    assert response.headers["x-content-type-options"] == "nosniff"
                    assert default.borrowed_tokens == POOL
                    assert not service.release.is_set()
                finally:
                    service.release.set()
                    await asyncio.gather(*providers, *others, *([capture] if capture else []))
                assert default.borrowed_tokens == 0
    asyncio.run(scenario())
