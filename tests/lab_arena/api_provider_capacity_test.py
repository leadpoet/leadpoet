"""Real ASGI routes must share the worker budget without provider starvation."""
from __future__ import annotations

import asyncio
import json
import threading

import anyio
import httpx
import pytest

from lab_arena.api import create_app
from lab_arena.service import ServiceError


FRAME = {"operation_id": "openrouter.chat", "parameters": {},
         "timeout_ms": 1000, "action_sequence": 1}
HEADERS = {"x-lab-arena-lease": "a" * 64}


class BlockingService:
    def __init__(self, default):
        self.default = default
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.calls = []
        self.active = 0
        self.maximum = 0
        self.maximum_default = 0
        self.other_started = 0

    def handle_provider(self, run_id, token, frame, cancel_requested):
        cancelled = cancel_requested()
        with self.lock:
            self.calls.append((run_id, token, frame, cancelled))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.maximum_default = max(self.maximum_default, self.default.borrowed_tokens)
        try:
            if not self.release.wait(5):
                raise AssertionError("test did not release blocked provider")
            if run_id == "fail":
                raise ServiceError("broker_unavailable", 503)
            return {"status": 200, "run_id": run_id}
        finally:
            with self.lock:
                self.active -= 1

    def signing_key_document(self):
        return {"key": "offline-test"}

    def public_current(self):
        return {"round": None}

    def handle_claim(self, envelope):
        return {"status": "no_pending", "envelope": envelope}

    def handle_complete(self, envelope):
        return {"status": "accepted", "envelope": envelope}

    def public_competition(self):
        with self.lock:
            self.other_started += 1
            self.maximum_default = max(self.maximum_default, self.default.borrowed_tokens)
        if not self.release.wait(5):
            raise AssertionError("test did not release other worker")
        return {"status": "ok"}


async def wait_until(predicate):
    with anyio.fail_after(3):
        while not predicate():
            await anyio.sleep(0.005)


def post_provider(client, run_id):
    return asyncio.create_task(client.post(
        "/arena/v1/runs/" + run_id + "/provider", json=FRAME, headers=HEADERS,
    ))


def test_provider_saturation_leaves_public_claim_and_completion_workers():
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        assert default.total_tokens == 40
        service = BlockingService(default)
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            providers = [post_provider(client, str(i)) for i in range(40)]
            others = []
            try:
                await wait_until(lambda: len(service.calls) >= 32)
                probes = await asyncio.wait_for(asyncio.gather(
                    client.get("/arena/v1/signing-key"), client.get("/arena/v1/current"),
                    client.post("/arena/v1/runs/claim", json={"claim": "unchanged"}),
                    client.post("/arena/v1/runs/done/complete", json={"body": {"run_id": "done"}}),
                ), 1)
                assert all(r.status_code == 200 for r in probes)
                assert len(service.calls) == service.active == 32
                assert default.borrowed_tokens == 32
                # Other work can use the remaining eight, without increasing
                # the process-wide default worker budget.
                others = [asyncio.create_task(client.get("/arena/v1/competition")) for _ in range(16)]
                await wait_until(lambda: service.other_started >= 8)
                assert service.other_started == 8
                assert default.borrowed_tokens == default.total_tokens == 40
            finally:
                service.release.set()
                responses = await asyncio.wait_for(asyncio.gather(*providers, *others), 3)
            assert all(r.status_code == 200 for r in responses)
            assert len(service.calls) == 40
            assert service.maximum == 32 and service.maximum_default <= 40
            assert all(token == "a" * 64 and frame == FRAME and cancelled is False
                       for _, token, frame, cancelled in service.calls)
            assert service.active == default.borrowed_tokens == 0
    asyncio.run(scenario())


def test_queued_provider_cancellation_and_failure_release_capacity():
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = BlockingService(default)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
            active = [post_provider(client, "fail" if i == 0 else str(i)) for i in range(32)]
            queued = None
            try:
                await wait_until(lambda: service.active == 32)
                queued = post_provider(client, "cancelled-before-worker")
                await anyio.sleep(0.02)
                assert len(service.calls) == 32
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
                assert len(service.calls) == 32
            finally:
                service.release.set()
                responses = await asyncio.wait_for(asyncio.gather(*active), 3)
            assert [r.status_code for r in responses].count(503) == 1
            assert [r.status_code for r in responses].count(200) == 31
            assert all(call[0] != "cancelled-before-worker" for call in service.calls)
            # A full second wave proves all sub-limiter tokens were returned,
            # including the token held by the failed request.
            service.release.clear()
            second = [post_provider(client, "next-" + str(i)) for i in range(32)]
            try:
                await wait_until(lambda: service.active == 32)
                assert default.borrowed_tokens == 32
            finally:
                service.release.set()
                assert all(r.status_code == 200 for r in await asyncio.gather(*second))
            assert service.active == default.borrowed_tokens == 0
    asyncio.run(scenario())


def test_provider_limit_is_per_app_and_under_capacity_arguments_are_unchanged():
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        services = [BlockingService(default), BlockingService(default)]
        clients = [httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(s)), base_url="http://test") for s in services]
        tasks = []
        try:
            tasks = [post_provider(clients[i], str(j)) for i in range(2) for j in range(20)]
            await wait_until(lambda: sum(s.active for s in services) == 40)
            assert [s.active for s in services] == [20, 20]
            assert default.borrowed_tokens == default.total_tokens == 40
        finally:
            for service in services:
                service.release.set()
            if tasks:
                responses = await asyncio.wait_for(asyncio.gather(*tasks), 3)
                assert all(r.status_code == 200 for r in responses)
            for client in clients:
                await client.aclose()
        assert default.borrowed_tokens == 0
        assert all(s.maximum == 20 and all(token == "a" * 64 and frame == FRAME and cancelled is False
                   for _, token, frame, cancelled in s.calls) for s in services)
    asyncio.run(scenario())


@pytest.mark.parametrize("queue", ["provider", "shared_worker"])
@pytest.mark.parametrize("cancel_task", [False, True])
def test_queued_provider_disconnect_or_cancellation_never_enters_service(queue, cancel_task):
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = BlockingService(default)
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            if queue == "provider":
                blockers = [post_provider(client, str(i)) for i in range(32)]
                await wait_until(lambda: service.active == 32)
            else:
                blockers = [asyncio.create_task(client.get("/arena/v1/competition")) for _ in range(40)]
                await wait_until(lambda: service.other_started == 40)
            body_read = asyncio.Event()
            messages = asyncio.Queue()
            sent = []
            raw = json.dumps(FRAME).encode()

            async def receive():
                if not body_read.is_set():
                    body_read.set()
                    return {"type": "http.request", "body": raw, "more_body": False}
                return await messages.get()

            async def send(message):
                sent.append(message)

            scope = {
                "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                "method": "POST", "scheme": "http", "path": "/arena/v1/runs/disconnected/provider",
                "raw_path": b"/arena/v1/runs/disconnected/provider", "query_string": b"",
                "headers": [(b"content-type", b"application/json"),
                            (b"x-lab-arena-lease", b"a" * 64)],
                "client": ("127.0.0.1", 1234), "server": ("test", 80),
            }
            request = asyncio.create_task(app(scope, receive, send))
            try:
                await body_read.wait()
                await anyio.sleep(0.02)
                assert not request.done()
                if cancel_task:
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                else:
                    messages.put_nowait({"type": "http.disconnect"})
            finally:
                service.release.set()
                await asyncio.wait_for(asyncio.gather(*blockers, request, return_exceptions=True), 3)
            assert all(call[0] != "disconnected" for call in service.calls)
            if cancel_task:
                assert not sent
            else:
                assert next(m["status"] for m in sent if m["type"] == "http.response.start") == 499
            assert service.active == default.borrowed_tokens == 0
            # Cancellation after taking the provider permit but before a
            # shared worker must return that permit too.
            service.release.clear()
            second = [post_provider(client, "after-" + str(i)) for i in range(32)]
            try:
                await wait_until(lambda: service.active == 32)
            finally:
                service.release.set()
                assert all(r.status_code == 200 for r in await asyncio.gather(*second))
            assert default.borrowed_tokens == 0
    asyncio.run(scenario())
