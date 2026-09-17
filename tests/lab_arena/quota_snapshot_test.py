"""Lease-scoped quota snapshots cross API, worker, and harness unchanged."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, lab_arena_checkpoint, runner, runtime, scoring
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
                "limit": contracts.CALL_QUOTAS_PER_ICP["openrouter"],
                "used": openrouter_used,
                "remaining": (
                    contracts.CALL_QUOTAS_PER_ICP["openrouter"]
                    - openrouter_used
                ),
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
    assert state.trusted_quota_failure is False
    assert decoded(server.handle_frame(control_frame())) == {
        "error": "quota_unavailable"
    }
    assert state.action_sequence == 0
    assert state.refusals == 0
    assert state.calls == []
    assert state.trusted_quota_failure is True


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
    assert state.trusted_quota_failure is False


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
    assert state.trusted_quota_failure is True


def test_successful_authoritative_reread_clears_quota_failure(tmp_path):
    class RecoveringApi(QuotaApi):
        def __init__(self):
            super().__init__()
            self.documents = [RuntimeError("private secret"), snapshot()]

        def quota_usage(self, run_id, lease_token):
            self.calls.append((run_id, lease_token))
            document = self.documents.pop(0)
            if isinstance(document, Exception):
                raise document
            return document

    api = RecoveringApi()
    state = runner.RunState(
        lease={"run_id": "run-1"}, lease_token=LEASE_TOKEN
    )
    server = runner.WorkerSocketServer(tmp_path / "worker.sock", api, state)

    assert decoded(server.handle_frame(control_frame())) == {
        "error": "quota_unavailable"
    }
    assert state.trusted_quota_failure is True
    assert decoded(server.handle_frame(control_frame())) == snapshot()
    assert state.trusted_quota_failure is False


class _ExecutorCache:
    def __init__(self, root: Path, *, source: bool = False):
        self.root = root
        self.source = source

    @contextmanager
    def acquire(self, *_args, **_kwargs):
        if self.source:
            source = self.root / "source"
            dependencies = self.root / "dependencies"
            source.mkdir(parents=True, exist_ok=True)
            dependencies.mkdir(parents=True, exist_ok=True)
            yield source, dependencies
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            yield self.root


class _CompletionQuotaApi(QuotaApi):
    def __init__(self, document, *, completion_status="failed"):
        super().__init__(document)
        self.leases = [_execution_lease()]
        self.completion_status = completion_status
        self.completions = []

    def claim(self, _envelope):
        if not self.leases:
            return {"status": "no_pending"}
        return self.leases.pop(0)

    def complete(self, envelope):
        self.completions.append(envelope)
        return {"status": self.completion_status}

    def quota_usage(self, run_id, lease_token):
        self.calls.append((run_id, lease_token))
        document = (
            self.document.pop(0)
            if isinstance(self.document, list)
            else self.document
        )
        if isinstance(document, Exception):
            raise document
        return document


class _StagedQuotaPreflightRuntime:
    def __init__(self, *, reads=1, output_bytes=None):
        self.reads = reads
        self.output_bytes = output_bytes
        self.outcomes = []

    def run_icp(self, spec, **_kwargs):
        assert spec.checkpoint_module_path is not None
        module_spec = importlib.util.spec_from_file_location(
            "staged_lab_arena_checkpoint", spec.checkpoint_module_path
        )
        assert module_spec is not None and module_spec.loader is not None
        staged_checkpoint = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(staged_checkpoint)
        previous = os.environ.get(staged_checkpoint.WORKER_SOCKET_ENV)
        os.environ[staged_checkpoint.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            for _ in range(self.reads):
                try:
                    self.outcomes.append(staged_checkpoint.quota_usage())
                except staged_checkpoint.QuotaUnavailable:
                    self.outcomes.append("quota_unavailable")
        finally:
            if previous is None:
                os.environ.pop(staged_checkpoint.WORKER_SOCKET_ENV, None)
            else:
                os.environ[staged_checkpoint.WORKER_SOCKET_ENV] = previous
        return runtime.fake_result(output_bytes=self.output_bytes)


def _execution_lease():
    digest = "sha256:" + "a" * 64
    return {
        "status": "leased",
        "round_id": "arena-2026-09-16",
        "run_id": "run-1",
        "assignment_id": "assignment-1",
        "submission_id": "submission-1",
        "source_ref": "source-1",
        "source_size_bytes": 1,
        "lease_token": LEASE_TOKEN,
        "kind": "execute",
        "image_digest": digest,
        "image_reference": "registry.example/arena@" + digest,
        "evaluation_date": "2026-09-16",
        "icp": {"max_companies": 5},
    }


def _run_preflight(tmp_path, api, sandbox):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with tempfile.TemporaryDirectory(prefix="quota-e2e-", dir="/tmp") as sockets:
        config = runner.RunnerConfig(
            round_id="arena-2026-09-16",
            identity=runner.RunnerIdentity(
                hotkey="5" * 48, sign=lambda _message: "signature"
            ),
            api=api,
            sandbox_runtime=sandbox,
            image_cache=_ExecutorCache(tmp_path / "image"),
            source_cache=_ExecutorCache(tmp_path / "source", source=True),
            work_dir=work_dir,
            socket_root=Path(sockets),
            clock=lambda: datetime(2026, 9, 16, tzinfo=timezone.utc),
        )
        arena_runner = runner.Runner(config)
        try:
            assert arena_runner.run_once() == 1
            return arena_runner.completed
        finally:
            arena_runner.close()


@pytest.mark.parametrize(
    "quota_document",
    [
        RuntimeError("private secret"),
        {**snapshot(), "private_field": "must fail closed"},
    ],
)
def test_authoritative_quota_failure_without_output_completes_as_provider_error(
    tmp_path, quota_document
):
    api = _CompletionQuotaApi(quota_document)
    sandbox = _StagedQuotaPreflightRuntime()

    completed = _run_preflight(tmp_path, api, sandbox)

    assert sandbox.outcomes == ["quota_unavailable"]
    assert completed == [{"run_id": "run-1", "result": {"status": "failed"}}]
    assert len(api.completions) == 1
    body = api.completions[0]["body"]
    assert body["output"] is None
    result = contracts.validate_run_result(body["result"])
    assert result["terminal_status"] == "provider_error"
    assert result["resource_summary"]["provider_call_count"] == 0
    assert result["failure_diagnostic"] == {
        "stage": "provider_call",
        "error_class": "provider_unavailable",
        "reason": "provider_error",
    }
    with pytest.raises(contracts.ArenaContractError, match="infrastructure reason"):
        scoring.build_scoring_plan(
            round_id="arena-2026-09-16",
            stage=1,
            runs=[
                {
                    "run_id": "run-1",
                    "submission_id": "submission-1",
                    "stage": 1,
                    "icp_position": 0,
                    "attempt": 1,
                    "status": "failed",
                    "terminal_cause": result["terminal_status"],
                }
            ],
        )


def test_quota_read_cap_is_client_misuse_and_stays_model_error(tmp_path):
    api = _CompletionQuotaApi(snapshot())
    sandbox = _StagedQuotaPreflightRuntime(
        reads=runner.MAX_QUOTA_SNAPSHOT_REQUESTS + 1
    )

    _run_preflight(tmp_path, api, sandbox)

    assert sandbox.outcomes[-1] == "quota_unavailable"
    assert len(api.calls) == 1
    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == "model_error"
    assert result["resource_summary"]["provider_call_count"] == 0


def test_successful_quota_reread_clears_completion_provenance(tmp_path):
    api = _CompletionQuotaApi(
        [RuntimeError("private secret"), snapshot()]
    )
    sandbox = _StagedQuotaPreflightRuntime(reads=2)

    _run_preflight(tmp_path, api, sandbox)

    assert sandbox.outcomes == ["quota_unavailable", snapshot()]
    assert len(api.calls) == 2
    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == "model_error"
    assert "failure_diagnostic" not in result


def test_valid_output_remains_accepted_after_authoritative_quota_failure(tmp_path):
    api = _CompletionQuotaApi(
        RuntimeError("private secret"), completion_status="accepted"
    )
    sandbox = _StagedQuotaPreflightRuntime(output_bytes=b'{"companies":[]}')

    completed = _run_preflight(tmp_path, api, sandbox)

    body = api.completions[0]["body"]
    assert completed == [{"run_id": "run-1", "result": {"status": "accepted"}}]
    assert sandbox.outcomes == ["quota_unavailable"]
    assert body["result"]["terminal_status"] == "accepted"
    assert "failure_diagnostic" not in body["result"]
    assert body["output"]["companies"] == []


def test_stale_quota_failure_completion_stays_stale_without_retry(tmp_path):
    api = _CompletionQuotaApi(
        RuntimeError("private secret"), completion_status="stale"
    )
    sandbox = _StagedQuotaPreflightRuntime()

    completed = _run_preflight(tmp_path, api, sandbox)

    assert completed == [{"run_id": "run-1", "result": {"status": "stale"}}]
    assert len(api.completions) == 1
    assert (
        api.completions[0]["body"]["result"]["terminal_status"]
        == "provider_error"
    )
