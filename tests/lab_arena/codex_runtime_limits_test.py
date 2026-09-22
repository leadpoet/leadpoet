"""Codex requests survive HTTP framing without escaping budget or depth limits."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import broker as br, lab_arena_codex as codex, operations as ops
from lab_arena.api import create_app
from lab_arena.service import ArenaService
from tests.lab_arena.codex_runtime_test import broker_socket, response
from tests.lab_arena.test_lab_arena_api import StubService
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeLedgerStore, FakeTransport, make_broker, price_table


def parameters(cap=16_384):
    schema = {"type": "string"}
    for _ in range(6):
        schema = {"type": "object", "properties": {"child": schema}}
    return {
        "model": "openai/gpt-4o-mini", "input": "Use the local tool.",
        "tools": [{"type": "namespace", "name": "functions", "description": "Local tools", "tools": [
            {"type": "function", "name": "inspect", "parameters": schema},
        ]}],
        "max_output_tokens": cap,
    }


@pytest.mark.parametrize("cap", [4096, 16_384, 32_768])
def test_gateway_admits_native_depth_and_reserves_the_explicit_output_cap(cap):
    broker, store, transport = make_broker(transport=FakeTransport([(200, response())]))

    class Service(StubService):
        handle_provider = ArenaService.handle_provider

        def _run_context(self, run_id, token):
            self.calls["context"] = (run_id, token)
            return {"round_id": CONTEXT.round_id}, CONTEXT

        def _broker_for(self, round_id):
            return broker

    service = Service()
    frame = {"operation_id": "openrouter.responses", "parameters": parameters(cap), "timeout_ms": 120_000, "action_sequence": 1}
    with TestClient(create_app(service)) as client:
        reply = client.post("/arena/v1/runs/run-1/provider", headers={"x-lab-arena-lease": "a" * 64}, json=frame)
        assert reply.status_code == 200, reply.text
        assert reply.json()["status"] == 200
        sent = json.loads(transport.sent[0]["body"])
        assert sent["max_output_tokens"] == cap
        call = next(iter(store.calls.values()))
        assert call["call_doc"]["max_output_tokens"] == cap
        assert call["amount"] == br.max_openrouter_cost_microusd(price_table(), sent["model"], br.normalized_request("openrouter.responses", parameters(cap)), max_output_tokens=cap)
        assert call["actual"] == 12
        # Another operation does not inherit the Responses depth allowance.
        service.calls.clear()
        reply = client.post("/arena/v1/runs/run-1/provider", headers={"x-lab-arena-lease": "a" * 64}, json={**frame, "operation_id": "openrouter.chat"})
        assert reply.status_code == 400
        assert "context" not in service.calls
        assert len(transport.sent) == 1


@pytest.mark.parametrize(
    ("requested_timeout_ms", "expected_timeout_seconds"),
    ((450_000, 450), (900_000, 600)),
)
def test_responses_transport_accepts_slow_success_once_with_a_600_second_cap(
    monkeypatch, requested_timeout_ms, expected_timeout_seconds,
):
    clock = [1000.0]

    class SlowSuccessTransport(FakeTransport):
        def send(self, **kwargs):
            result = super().send(**kwargs)
            clock[0] += 301.0
            return result

    monkeypatch.setattr(br.time, "monotonic", lambda: clock[0])
    transport = SlowSuccessTransport([(200, response())])
    broker, store, transport = make_broker(transport=transport)

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.responses",
        parameters=parameters(),
        timeout_ms=requested_timeout_ms,
        action_sequence=1,
    )

    assert result.status == 200
    assert len(transport.sent) == len(store.calls) == 1
    assert transport.sent[0]["timeout"] == pytest.approx(
        expected_timeout_seconds
    )
    assert next(iter(store.calls.values()))["kind"] == "settlement"


def test_larger_reasoning_allowance_cannot_overdraw_the_budget():
    params = br.normalized_request("openrouter.responses", parameters())
    needed = br.max_openrouter_cost_microusd(price_table(), params["model"], params, max_output_tokens=params["max_output_tokens"])
    broker, store, transport = make_broker(store=FakeLedgerStore(openrouter_capacity=needed - 1))
    result = broker.execute(CONTEXT, operation_id="openrouter.responses", parameters=params, timeout_ms=120_000, action_sequence=1)
    assert result.status >= 400
    assert not transport.sent
    assert next(iter(store.calls.values()))["reason"] == "provider_cost_cap"
    with pytest.raises(ops.OperationRequestError):
        br.normalized_request("openrouter.chat", {"model": params["model"], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4097})


@pytest.mark.parametrize("cap", [True, 0, 32_769])
def test_invalid_bridge_output_cap_is_rejected_before_listening(cap):
    with pytest.raises(codex.CodexRuntimeError, match="output token"):
        codex.ResponsesBridge("/unused.sock", max_output_tokens=cap)


def test_bridge_enforces_its_session_allowance(monkeypatch):
    with broker_socket(monkeypatch, FakeTransport([(200, response())])) as (store, transport, path), codex.ResponsesBridge(str(path), max_output_tokens=8000) as bridge:
        with httpx.Client(trust_env=False) as client:
            kwargs = {"headers": {"Authorization": "Bearer " + bridge.token}}
            reply = client.post(bridge.base_url + "/responses", json={"model": "openai/gpt-4o-mini", "input": "hi"}, **kwargs)
            assert reply.status_code == 200
            assert json.loads(transport.sent[0]["body"])["max_output_tokens"] == 8000
            reply = client.post(bridge.base_url + "/responses", json={"model": "openai/gpt-4o-mini", "input": "hi", "max_output_tokens": 8001}, **kwargs)
            assert reply.status_code == 400
            assert len(transport.sent) == len(store.calls) == 1


@pytest.mark.parametrize("payload", [
    b"not-json", b"[]", b"{}",
    json.dumps(response(output=[{"type": "message", "content": [{"type": "output_text"}]}])).encode(),
    json.dumps(response(output=[{"type": "message", "content": [{"type": "output_text", "text": 3}]}])).encode(),
])
def test_malformed_upstream_reply_has_a_bounded_gateway_error(monkeypatch, payload):
    monkeypatch.setattr(codex, "_dispatch", lambda *args, **kwargs: (200, payload))
    with codex.ResponsesBridge("/unused.sock") as bridge, httpx.Client(trust_env=False) as client:
        reply = client.post(bridge.base_url + "/responses", headers={"Authorization": "Bearer " + bridge.token},
                            json={"model": "openai/gpt-4o-mini", "input": "hi", "stream": True})
    assert reply.status_code == 502
    assert reply.json() == {"error": {"message": "Arena broker unavailable"}}
