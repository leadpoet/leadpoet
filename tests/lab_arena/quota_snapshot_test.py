"""Lease-scoped quota snapshots cross API, worker, and harness unchanged."""

from __future__ import annotations

import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, lab_arena_checkpoint, runner
from lab_arena.api import create_app
from lab_arena.service import ServiceError


LEASE_TOKEN = "a" * 64


def snapshot(*, openrouter_used: int = 8, openrouter_inflight: int = 2):
    return {
        "schema_version": lab_arena_checkpoint.QUOTA_SNAPSHOT_SCHEMA_VERSION,
        "providers": {
            "scrapingdog": {
                "limit": 30,
                "used": 0,
                "remaining": 30,
                "inflight": 0,
            },
            "deepline": {
                "limit": 30,
                "used": 1,
                "remaining": 29,
                "inflight": 0,
            },
            "openrouter": {
                "limit": 60,
                "used": openrouter_used,
                "remaining": 60 - openrouter_used,
                "inflight": openrouter_inflight,
            },
        },
    }


class QuotaService:
    def __init__(self, *, failure: bool = False):
        self.failure = failure
        self.calls = []

    def handle_quota_snapshot(self, run_id, lease_token):
        self.calls.append((run_id, lease_token))
        if self.failure:
            raise ServiceError("quota_unavailable", 503)
        return snapshot()


def test_authenticated_quota_api_returns_only_bounded_counters():
    service = QuotaService()
    with TestClient(create_app(service)) as client:
        response = client.get(
            "/arena/v1/runs/run-1/quota",
            headers={"x-lab-arena-lease": LEASE_TOKEN},
        )
        denied = client.get(
            "/arena/v1/runs/run-1/quota",
            headers={"x-lab-arena-lease": "wrong"},
        )

    assert response.status_code == 200
    assert response.json() == snapshot()
    assert response.headers["cache-control"] == "no-store"
    assert service.calls == [("run-1", LEASE_TOKEN)]
    assert denied.status_code == 401
    encoded = response.text.lower()
    for forbidden in (
        "run-1",
        LEASE_TOKEN,
        "credential",
        "account",
        "microusd",
        "call_identity",
    ):
        assert forbidden.lower() not in encoded


def test_quota_api_failure_is_generic_and_no_store():
    with TestClient(create_app(QuotaService(failure=True))) as client:
        response = client.get(
            "/arena/v1/runs/run-1/quota",
            headers={"x-lab-arena-lease": LEASE_TOKEN},
        )
    assert response.status_code == 503
    assert response.json() == {
        "status": "rejected",
        "code": "quota_unavailable",
    }
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, json={"detail": "private database diagnostic"}),
        httpx.Response(200, json={**snapshot(), "run_id": "private-run"}),
        httpx.Response(
            200,
            json={
                **snapshot(),
                "providers": {
                    **snapshot()["providers"],
                    "openrouter": {
                        **snapshot()["providers"]["openrouter"],
                        "remaining": 59,
                    },
                },
            },
        ),
    ],
)
def test_api_client_fails_closed_on_http_or_schema_error(response):
    observed = []

    def handler(request):
        observed.append(request)
        return response

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = runner.HttpArenaApiClient("https://arena.example", client=http)
        with pytest.raises(runner.RunnerError, match="run quota is unavailable") as caught:
            client.quota_usage("run-1", LEASE_TOKEN)
    assert len(observed) == 1
    assert observed[0].url.path == "/arena/v1/runs/run-1/quota"
    assert observed[0].headers["x-lab-arena-lease"] == LEASE_TOKEN
    assert "private" not in str(caught.value)


def test_api_client_accepts_exact_quota_document():
    def handler(request):
        return httpx.Response(200, json=snapshot())

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = runner.HttpArenaApiClient("https://arena.example", client=http)
        assert client.quota_usage("run-1", LEASE_TOKEN) == snapshot()


class QuotaApi:
    def __init__(self, document=None):
        self.document = document if document is not None else snapshot()
        self.calls = []

    def quota_usage(self, run_id, lease_token):
        self.calls.append((run_id, lease_token))
        if isinstance(self.document, Exception):
            raise self.document
        return self.document


def control_frame(**extra):
    return contracts.canonical_json(
        {**lab_arena_checkpoint.QUOTA_CONTROL_FRAME, **extra}
    ).encode("utf-8")


def decoded(payload):
    return json.loads(payload.decode("utf-8"))


def test_sdk_worker_full_read_is_separate_from_provider_counters(monkeypatch):
    api = QuotaApi()
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    with tempfile.TemporaryDirectory(prefix="quota-", dir="/tmp") as directory:
        socket_path = Path(directory) / "worker.sock"
        server = runner.WorkerSocketServer(socket_path, api, state)
        server.start()
        monkeypatch.setenv(
            lab_arena_checkpoint.WORKER_SOCKET_ENV, str(socket_path)
        )
        try:
            assert lab_arena_checkpoint.quota_usage() == snapshot()
        finally:
            server.stop()

    assert api.calls == [("run-1", LEASE_TOKEN)]
    assert state.action_sequence == 0
    assert state.refusals == 0
    assert state.calls == []


def test_control_frame_denies_extra_fields_and_malformed_api_result(tmp_path):
    malformed = {**snapshot(), "lease_token": "secret"}
    api = QuotaApi(malformed)
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", api, state)

    assert decoded(server.handle_frame(control_frame(extra=True))) == {
        "error": "invalid_frame"
    }
    assert decoded(
        server.handle_frame(
            contracts.canonical_json(
                {
                    "schema_version": (
                        lab_arena_checkpoint.QUOTA_CONTROL_SCHEMA_VERSION
                    )
                }
            ).encode("utf-8")
        )
    ) == {"error": "invalid_frame"}
    assert decoded(server.handle_frame(b"{")) == {"error": "invalid_frame"}
    assert state.quota_request_count == 0
    assert decoded(server.handle_frame(control_frame())) == {
        "error": "quota_unavailable"
    }
    assert state.action_sequence == 0
    assert state.refusals == 0
    assert state.calls == []


def test_quota_reads_are_one_hz_cached_and_capped_without_counter_changes(
    tmp_path,
):
    clock = [10.0]
    api = QuotaApi()
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    server = runner.WorkerSocketServer(
        tmp_path / "worker.sock", api, state, monotonic=lambda: clock[0]
    )

    assert decoded(server.handle_frame(control_frame())) == snapshot()
    assert decoded(server.handle_frame(control_frame())) == snapshot()
    assert len(api.calls) == 1
    clock[0] += 1.0
    assert decoded(server.handle_frame(control_frame())) == snapshot()
    assert len(api.calls) == 2
    for _ in range(runner.MAX_QUOTA_SNAPSHOT_REQUESTS - 3):
        assert decoded(server.handle_frame(control_frame())) == snapshot()
    assert decoded(server.handle_frame(control_frame())) == {
        "error": "quota_unavailable"
    }
    assert state.quota_request_count == runner.MAX_QUOTA_SNAPSHOT_REQUESTS
    assert state.action_sequence == 0
    assert state.refusals == 0
    assert state.calls == []


def test_concurrent_quota_reads_singleflight(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingApi(QuotaApi):
        def quota_usage(self, run_id, lease_token):
            self.calls.append((run_id, lease_token))
            entered.set()
            assert release.wait(2)
            return self.document

    api = BlockingApi()
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", api, state)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(server._quota_snapshot) for _ in range(8)]
        assert entered.wait(1)
        release.set()
        results = [future.result(timeout=2) for future in futures]

    assert results == [snapshot()] * 8
    assert api.calls == [("run-1", LEASE_TOKEN)]
    assert state.action_sequence == 0
    assert state.refusals == 0
    assert state.calls == []


def test_sdk_failure_is_one_generic_exception(monkeypatch):
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    with tempfile.TemporaryDirectory(prefix="quota-", dir="/tmp") as directory:
        socket_path = Path(directory) / "worker.sock"
        server = runner.WorkerSocketServer(
            socket_path, QuotaApi(RuntimeError("private secret")), state
        )
        server.start()
        monkeypatch.setenv(
            lab_arena_checkpoint.WORKER_SOCKET_ENV, str(socket_path)
        )
        try:
            with pytest.raises(
                lab_arena_checkpoint.QuotaUnavailable,
                match="^quota unavailable$",
            ):
                lab_arena_checkpoint.quota_usage()
        finally:
            server.stop()
