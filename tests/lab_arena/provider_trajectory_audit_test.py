"""Focused provider retry and trajectory persistence coverage."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import broker as broker_module
from lab_arena.broker import BrokerResult, RunContext
from lab_arena.store import (
    ArenaStore,
    ArenaStoreError,
    ArenaStoreUnavailable,
    PostgrestTransport,
)
from tests.lab_arena.trajectory_test import _run, _service_for_trajectory


FRAME = {
    "operation_id": "openrouter.chat",
    "parameters": {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    },
    "timeout_ms": 120_000,
    "action_sequence": 12,
}


def _result(*, provider_attempt: int, error: bool) -> BrokerResult:
    call_identity = "sha256:" + str(provider_attempt) * 64
    call = {
        "call_identity": call_identity,
        "base_call_identity": "sha256:" + "a" * 64,
        "operation_id": "openrouter.chat",
        "provider": "openrouter",
        "provider_status": 401 if error else 200,
        "provider_attempt": provider_attempt,
        "funding_source": "miner_key",
    }
    if error:
        call["error_code"] = "miner_credentials_unavailable"
    return BrokerResult(
        503 if error else 200,
        {},
        b'{"error":"unavailable"}' if error else b'{"answer":"ok"}',
        call,
    )


def _retrying_broker(monkeypatch, *, summary_failure: bool = False):
    provider = broker_module.Broker.__new__(broker_module.Broker)
    provider._retry_miner_credential_for = lambda _context: True
    provider._mark_provider_fallback = None
    results = iter((_result(provider_attempt=1, error=True), _result(provider_attempt=2, error=False)))
    provider._execute_once = lambda *_args, **_kwargs: next(results)
    if summary_failure:
        monkeypatch.setattr(
            broker_module,
            "_provider_attempt_summary",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("observer")),
        )
    return provider


def _context() -> RunContext:
    return RunContext(
        run_id="run-1",
        assignment_id="assignment-1",
        icp_position=0,
        lease_token_hash="sha256:" + "b" * 64,
        miner_hotkey="5" + "a" * 47,
        submission_id="submission-1",
        stage=1,
        round_id="arena-2026-09-26",
    )


def test_champion_retry_trace_is_private_bounded_and_keeps_worker_document(monkeypatch):
    provider = _retrying_broker(monkeypatch)

    result = provider.execute(
        _context(),
        operation_id=FRAME["operation_id"],
        parameters=FRAME["parameters"],
        action_sequence=FRAME["action_sequence"],
        timeout_ms=FRAME["timeout_ms"],
    )

    assert result.status == 200
    assert len(result.attempt_trace) == 2
    assert [item["provider_attempt"] for item in result.attempt_trace] == [1, 2]
    assert [item["provider_status"] for item in result.attempt_trace] == [401, 200]
    assert result.attempt_trace[0]["error_code"] == "miner_credentials_unavailable"
    for item in result.attempt_trace:
        assert set(item) <= {
            "occurred_at", "elapsed_ms", "http_status", "provider_attempt",
            "call_identity", "operation_id", "provider", "error_code",
            "provider_status",
        }
        assert datetime.fromisoformat(item["occurred_at"].replace("Z", "+00:00")).tzinfo
    worker_document = result.to_document()
    assert set(worker_document) == {"status", "headers", "body_b64", "call"}
    assert "attempt_trace" not in json.dumps(worker_document)


def test_attempt_summary_failure_does_not_change_provider_retry_result(monkeypatch):
    provider = _retrying_broker(monkeypatch, summary_failure=True)

    result = provider.execute(
        _context(),
        operation_id=FRAME["operation_id"],
        parameters=FRAME["parameters"],
        action_sequence=FRAME["action_sequence"],
        timeout_ms=FRAME["timeout_ms"],
    )

    assert result.status == 200
    assert result.body == b'{"answer":"ok"}'
    assert result.attempt_trace == ()


def test_provider_trajectory_retries_first_transport_failure_with_same_uuid():
    class Store:
        calls = []
        persisted = []
        get_run = staticmethod(lambda _run_id: _run())

        @classmethod
        def append_trajectory_events(cls, _run_id, _lease_hash, events):
            cls.calls.append([dict(item) for item in events])
            if len(cls.calls) == 1:
                raise ArenaStoreUnavailable("transport")
            cls.persisted.extend(events)
            return {
                "status": "accepted", "accepted": len(events),
                "inserted": len(events), "existing": 0,
            }

    expected = _result(provider_attempt=1, error=False)
    service = _service_for_trajectory(
        Store(), SimpleNamespace(execute=lambda *_args, **_kwargs: expected)
    )

    assert service.handle_provider("run-1", "lease-token", FRAME) == expected.to_document()
    assert Store.calls[0][0]["event_id"] == Store.calls[1][0]["event_id"]
    assert [item["kind"] for item in Store.persisted] == [
        "provider.request", "provider.response",
    ]


def test_provider_trajectory_replay_is_idempotent_after_committed_response_loss():
    class Store:
        calls = []
        persisted = {}
        get_run = staticmethod(lambda _run_id: _run())

        @classmethod
        def append_trajectory_events(cls, _run_id, _lease_hash, events):
            event = dict(events[0])
            cls.calls.append(event)
            existed = event["event_id"] in cls.persisted
            cls.persisted.setdefault(event["event_id"], event)
            if len(cls.calls) == 1:
                raise ArenaStoreUnavailable("response lost after commit")
            return {
                "status": "accepted", "accepted": 1,
                "inserted": 0 if existed else 1,
                "existing": 1 if existed else 0,
            }

    expected = _result(provider_attempt=1, error=False)
    service = _service_for_trajectory(
        Store(), SimpleNamespace(execute=lambda *_args, **_kwargs: expected)
    )

    assert service.handle_provider("run-1", "lease-token", FRAME) == expected.to_document()
    assert Store.calls[0]["event_id"] == Store.calls[1]["event_id"]
    assert len(Store.persisted) == 2
    assert sorted(item["kind"] for item in Store.persisted.values()) == [
        "provider.request", "provider.response",
    ]


def test_provider_trajectory_does_not_retry_nontransport_denial():
    class Store:
        calls = 0
        get_run = staticmethod(lambda _run_id: _run())

        @classmethod
        def append_trajectory_events(cls, _run_id, _lease_hash, _events):
            cls.calls += 1
            raise ArenaStoreError("trajectory event limit")

    expected = _result(provider_attempt=1, error=False)
    service = _service_for_trajectory(
        Store(), SimpleNamespace(execute=lambda *_args, **_kwargs: expected)
    )

    assert service.handle_provider("run-1", "lease-token", FRAME) == expected.to_document()
    assert Store.calls == 2  # One request and one response, with no denial replay.


@pytest.mark.parametrize("first_reply", [502, 503, 504, "invalid_json"])
def test_postgrest_trajectory_availability_retries_same_uuid(first_reply):
    rpc_events = []
    persisted = {}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=[_run()])
        payload = json.loads(request.content)
        event = payload["p_events"][0]
        rpc_events.append(event)
        if len(rpc_events) == 1:
            if first_reply == "invalid_json":
                persisted[event["event_id"]] = event
                return httpx.Response(200, content=b"not-json")
            return httpx.Response(
                first_reply, json={"message": "upstream unavailable"}
            )
        existed = event["event_id"] in persisted
        persisted.setdefault(event["event_id"], event)
        return httpx.Response(
            200,
            json={
                "status": "accepted", "accepted": 1,
                "inserted": 0 if existed else 1,
                "existing": 1 if existed else 0,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        store = ArenaStore(PostgrestTransport(
            "https://database.example",
            service_key="sb_secret_scoped-test",
            http_client=http,
        ))
        expected = _result(provider_attempt=1, error=False)
        service = _service_for_trajectory(
            store, SimpleNamespace(execute=lambda *_args, **_kwargs: expected)
        )

        assert service.handle_provider(
            "run-1", "lease-token", FRAME
        ) == expected.to_document()

    assert rpc_events[0]["event_id"] == rpc_events[1]["event_id"]
    assert len({event["event_id"] for event in rpc_events}) == 2
    assert sorted(event["kind"] for event in persisted.values()) == [
        "provider.request", "provider.response",
    ]


def test_postgrest_trajectory_contract_denial_and_other_rpc_are_not_retried():
    trajectory_calls = 0

    def handler(request):
        nonlocal trajectory_calls
        if request.method == "GET":
            return httpx.Response(200, json=[_run()])
        if request.url.path.endswith("/lab_arena_append_trajectory_events_v1"):
            trajectory_calls += 1
            return httpx.Response(
                400,
                json={
                    "code": "54000",
                    "message": "lab_arena_trajectory_provider_limit",
                },
            )
        return httpx.Response(503, json={"message": "unavailable"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        transport = PostgrestTransport(
            "https://database.example",
            service_key="sb_secret_scoped-test",
            http_client=http,
        )
        store = ArenaStore(transport)
        expected = _result(provider_attempt=1, error=False)
        service = _service_for_trajectory(
            store, SimpleNamespace(execute=lambda *_args, **_kwargs: expected)
        )

        assert service.handle_provider(
            "run-1", "lease-token", FRAME
        ) == expected.to_document()
        with pytest.raises(ArenaStoreError) as caught:
            transport.rpc(
                "lab_arena_run_quota_snapshot_v1",
                {
                    "p_run_id": "run-1",
                    "p_lease_token_hash": "sha256:" + "a" * 64,
                },
            )

    assert trajectory_calls == 2  # One denied request and one denied response.
    assert type(caught.value) is ArenaStoreError


def test_retry_trace_fits_terminal_event_and_is_not_returned_to_worker(monkeypatch):
    attempts = tuple(
        {
            "occurred_at": "2026-09-26T12:00:00.000Z",
            "elapsed_ms": 3_600_000,
            "http_status": 503 if attempt < 4 else 200,
            "provider_status": 429 if attempt < 4 else 200,
            "provider_attempt": attempt,
            "call_identity": "sha256:" + str(attempt) * 64,
            "operation_id": "openrouter.responses",
            "provider": "openrouter",
            "error_code": "miner_credentials_unavailable",
        }
        for attempt in range(1, 5)
    )
    expected = BrokerResult(
        200,
        {},
        json.dumps({"output_text": "x" * 250_000}).encode(),
        {
            "call_identity": "sha256:" + "4" * 64,
            "operation_id": "openrouter.responses",
            "provider": "openrouter",
            "provider_status": 200,
            "provider_attempt": 4,
        },
        attempts,
    )

    class Store:
        persisted = []
        get_run = staticmethod(lambda _run_id: _run())

        @classmethod
        def append_trajectory_events(cls, _run_id, _lease_hash, events):
            cls.persisted.extend(events)
            return {
                "status": "accepted", "accepted": len(events),
                "inserted": len(events), "existing": 0,
            }

    service = _service_for_trajectory(
        Store(), SimpleNamespace(execute=lambda *_args, **_kwargs: expected)
    )

    returned = service.handle_provider("run-1", "lease-token", FRAME)
    assert returned == expected.to_document()
    terminal = Store.persisted[-1]
    assert len(terminal["content"]["provider_attempts"]) == 4
    assert len(json.dumps(terminal, separators=(",", ":")).encode()) < 8192
    assert "provider_attempts" not in json.dumps(returned)
