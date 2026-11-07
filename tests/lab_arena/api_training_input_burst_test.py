"""Bounded input bursts preserve evidence and drop timed-out queued work."""
from __future__ import annotations

import asyncio
import socket
import threading
from types import SimpleNamespace

import anyio
import httpx
import pytest
import uvicorn

from lab_arena import api as arena_api, runner, training_input
from lab_arena.api import create_app
from lab_arena.service import LocalObjectStore
from tests.lab_arena.api_training_input_capacity_test import (
    CAPTURE_LIMIT, CLAIM_HASH, TOKEN, CaptureService, post_input, real_service,
)
from tests.lab_arena.api_provider_capacity_test import wait_until


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("worker_limit", [2, 8])
def test_input_burst_bounds_storage_work_and_preserves_evidence(tmp_path, monkeypatch, kind, worker_limit):
    monkeypatch.setattr(arena_api, "TRAINING_INPUT_WORKER_LIMIT", worker_limit)

    async def scenario():
        objects = LocalObjectStore(tmp_path / "private-objects")
        original_put = objects.put
        release = threading.Event()
        lock = threading.Lock()
        entered = []

        def put(ref, data):
            with lock:
                entered.append(ref)
            assert release.wait(5), "test did not release bounded storage work"
            original_put(ref, data)

        objects.put = put
        app = create_app(real_service(objects, kind=kind))
        # Exercise the large-body/privacy/integer path in every worker. The
        # storage gate, rather than a latency estimate, proves concurrency.
        document = {"prompt": "é\n" * 300_000, "api_key": "synthetic-secret",
                    "schema": {"maximum": 2 ** 64 - 1}, "lease_echo": TOKEN}
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                captures = [post_input(client, "burst-" + str(i), document) for i in range(8)]
                try:
                    await wait_until(lambda: len(entered) == worker_limit)
                    assert not any(task.done() for task in captures)
                    assert anyio.to_thread.current_default_thread_limiter().borrowed_tokens == 0
                finally:
                    release.set()
                    responses = await asyncio.gather(*captures)
        assert all(response.status_code == 200 for response in responses)
        for index, response in enumerate(responses):
            run_id = "burst-" + str(index)
            expected, compressed = training_input.build_snapshot(
                run_id, kind, document, lease_generation=1, claim_request_hash=CLAIM_HASH, secrets=(TOKEN,),
            )
            assert objects.get(training_input.object_ref(run_id, 1)) == compressed
            assert training_input.read_input(objects, run_id, kind, 1, CLAIM_HASH) == expected
            assert response.json()["input_sha256"] == expected["input_sha256"]
            assert response.json()["privacy"] == {"redacted": True, "redactions": ["credential"]}
            assert response.headers["cache-control"] == "no-store"
        assert document["api_key"] == "synthetic-secret" and document["lease_echo"] == TOKEN
        assert document["schema"]["maximum"] == 2 ** 64 - 1
    asyncio.run(scenario())


def test_real_http_timeout_retries_drop_queued_inputs_and_leave_control_capacity():
    async def scenario():
        default = anyio.to_thread.current_default_thread_limiter()
        service = CaptureService(default)
        app = create_app(service)
        disconnected = []

        async def observed_app(scope, receive, send):
            try:
                await app(scope, receive, send)
            finally:
                if scope.get("path") == "/arena/v1/runs/timed-out/training-input":
                    disconnected.append(True)

        # Bind once and pass the socket to Uvicorn to avoid a free-port race.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            server = uvicorn.Server(uvicorn.Config(observed_app, log_level="critical", ws="none", lifespan="on"))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            blockers = []
            try:
                await wait_until(lambda: server.started)
                base_url = "http://127.0.0.1:%d" % listener.getsockname()[1]
                async with httpx.AsyncClient(base_url=base_url, timeout=3) as client:
                    blockers = [post_input(client, "active-" + str(i)) for i in range(CAPTURE_LIMIT)]
                    await wait_until(lambda: service.capture_active == CAPTURE_LIMIT)
                    attempts = []

                    def record():
                        with httpx.Client() as wire:
                            api = runner.HttpArenaApiClient(base_url, client=wire)

                            def send(run_id, token, document):
                                attempts.append(run_id)
                                # Shorten only this test's transport deadline;
                                # use the real runner client, retry and ASGI path.
                                return api.training_input(run_id, token, document, timeout_seconds=0.1)

                            return runner._record_training_input(
                                SimpleNamespace(api=SimpleNamespace(training_input=send)),
                                {"run_id": "timed-out", "kind": "score", "lease_generation": 1,
                                 "claim_request_hash": CLAIM_HASH}, TOKEN, {"input": "private"},
                            )

                    assert await anyio.to_thread.run_sync(record) is False
                    assert attempts == ["timed-out", "timed-out"]
                    await wait_until(lambda: len(disconnected) == 2)
                    assert service.capture_active == len(service.input_calls) == CAPTURE_LIMIT
                    control = await client.post("/arena/v1/runs/claim", json={})
                    assert control.status_code == 200
                    service.capture_release.set()
                    assert all(response.status_code == 200 for response in await asyncio.gather(*blockers))
                    # A successful fresh wave after both disconnected retries
                    # proves they did not retain or later consume worker slots.
                    assert (await post_input(client, "after-timeouts")).status_code == 200
                    assert all(row[0] != "timed-out" for row in service.input_calls)
                    assert service.capture_maximum == CAPTURE_LIMIT
                    assert default.borrowed_tokens == 0
            finally:
                service.capture_release.set()
                await asyncio.gather(*blockers, return_exceptions=True)
                server.should_exit = True
                await asyncio.wait_for(serving, 3)
    asyncio.run(scenario())
