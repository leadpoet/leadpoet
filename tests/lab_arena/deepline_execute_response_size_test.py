"""A paid Deepline JSON result near the worker/ledger bounds stays usable."""

import json
from dataclasses import replace

import pytest

from lab_arena import broker as br, contracts, operations, runner, shim
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeLedgerStore, make_broker
from tests.lab_arena.test_lab_arena_runner import lease


NATIVE_ID = "public-execute:11111111-2222-4333-8444-555555555555"
TOOL = "contextdev_post_web_search"
LIMIT = 2_097_152
PARAMETERS = {
    "tool": TOOL,
    "payload": {
        "query": "synthetic company evidence",
        "numResults": 10,
        "markdownOptions": {"enabled": True, "useMainContentOnly": True},
    },
}
CATALOG = frozen(row(
    TOOL, provider="contextdev", categories=["research"],
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "numResults": {"type": "integer", "minimum": 1, "maximum": 10},
            "markdownOptions": {"type": "object", "properties": {
                "enabled": {"type": "boolean"},
                "useMainContentOnly": {"type": "boolean"},
            }},
        },
        "required": ["query"],
    },
))


def _body_of_size(size):
    """Ten anonymous search results, one large markdown field, exact byte size."""

    document = {
        "job_id": NATIVE_ID,
        "status": "completed",
        "billing": {"credits_charged": 0.089, "pricing_status": "final"},
        "result": {"data": {"results": [
            {"url": "https://example.com/page", "title": "Synthetic result",
             "markdown": {"content": "" if index == 0 else "x" * 7000}}
            for index in range(10)
        ]}},
    }
    encode = lambda: json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    padding = size - len(encode())
    assert padding > 0
    document["result"]["data"]["results"][0]["markdown"]["content"] = "x" * padding
    body = encode()
    assert len(body) == size
    assert len(document["result"]["data"]["results"]) == 10
    assert len(document["result"]["data"]["results"][0]["markdown"]["content"]) > 900_000
    return body


class ResponseTransport:
    def __init__(self, body):
        self.body = body
        self.requests = []

    def send(self, **request):
        self.requests.append(request)
        if request["method"] == "POST":
            return br.ProviderResponse(200, {"content-type": "application/json"}, self.body)
        assert request["method"] == "GET"
        if "/executions/by-key/" in request["url"]:
            key = request["url"].split("/by-key/", 1)[1]
            from urllib.parse import unquote
            return br.ProviderResponse(200, {"x-deepline-idempotency-supported": "true"},
                json.dumps({"requestId": NATIVE_ID, "toolId": TOOL,
                    "executionRecovery": {"idempotencyKey": unquote(key), "state": "completed"}}).encode())
        assert request["url"] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE_ID, safe="")
        return br.ProviderResponse(200, {}, json.dumps({"recent": {
            "request_id": NATIVE_ID, "entries": [{
                "id": "synthetic-exact-charge", "request_id": NATIVE_ID,
                "provider": "contextdev", "operation": TOOL,
                "credits": "0.089", "delta": "-0.089",
                "charge_state": "posted", "charge_finality": "final",
                "billing_mode": "deduct_on_settle", "metadata": {},
            }]}}).encode())


def _execute(body):
    transport = ResponseTransport(body)
    broker, store, _ = make_broker(store=FakeLedgerStore(), transport=transport)
    context = replace(CONTEXT, round_id="arena-2026-10-18", deepline_catalog=CATALOG)
    request = dict(operation_id="deepline.execute", parameters=PARAMETERS,
                   action_sequence=0, timeout_ms=30_000)
    return broker, store, transport, context, request


@pytest.mark.parametrize("size", [1_059_913, 1_160_405, 1_479_442, LIMIT])
def test_paid_search_result_preserves_body_charge_and_worker_bounds(size, tmp_path):
    body = _body_of_size(size)
    broker, store, transport, context, request = _execute(body)
    first = broker.execute(context, **request)
    assert first.status == 200 and first.body == body
    assert first.call["outcome"] == "settled"
    assert first.call["actual_microusd"] == 8_900
    assert [item["method"] for item in transport.requests] == ["POST"]
    saved = store.calls[first.call["call_identity"]]
    assert saved["terminal"]["call_succeeded"] is True
    assert saved["actual"] == 8_900
    assert len(contracts.canonical_json(saved["terminal"]).encode()) < 4 * 1_048_576
    assert len(contracts.canonical_json(first.to_document()).encode()) < 4 * 1_048_576
    worker_frame = shim.encode_worker_response(first.status, first.headers, first.body)
    assert len(worker_frame) < shim.MAX_RESPONSE_FRAME_BYTES
    assert shim.parse_worker_response(json.loads(worker_frame))[2] == body
    class Api:
        def provider(self, _run_id, _lease_token, _frame):
            return first.to_document()
    worker_state = runner.RunState(lease={**lease("response-size"),
                                          "deepline_catalog": CATALOG}, lease_token="token")
    worker = runner.WorkerSocketServer(tmp_path / "worker.sock", Api(), worker_state)
    actual_frame = worker.handle_frame(shim.build_operation_frame(
        "deepline.execute", PARAMETERS, 30_000))
    assert len(actual_frame) < shim.MAX_RESPONSE_FRAME_BYTES
    assert shim.parse_worker_response(json.loads(actual_frame))[2] == body
    assert len(worker_state.calls) == 1

    again = broker.execute(context, **request)
    assert again.status == 200 and again.body == body
    assert again.call["idempotent"] is True
    assert again.call["actual_microusd"] == 8_900
    assert [item["method"] for item in transport.requests] == ["POST"]
    assert store.log.count("dispatch") == store.log.count("settle") == 1


def test_over_cap_result_still_fails_closed_and_settles_exact_charge():
    broker, store, transport, context, request = _execute(_body_of_size(LIMIT + 1))
    result = broker.execute(context, **request)
    assert result.status == 502 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 8_900
    assert result.call["failure_stage"] == "response_sanitization"
    assert result.call["error_class"] == "OperationResponseError"
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is False
    assert [item["method"] for item in transport.requests] == ["POST"]


def test_malformed_native_response_cannot_become_success_after_exact_readback(monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    broker, store, transport, context, request = _execute(b'{"status":"completed",')
    result = broker.execute(context, **request)
    assert result.status == 502 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 8_900
    assert result.call["failure_stage"] == "response_sanitization"
    assert result.call["error_class"] == "OperationResponseError"
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is False
    assert [item["method"] for item in transport.requests] == ["POST", "GET", "GET"]


def test_other_operation_limits_do_not_change():
    assert operations.get_operation("deepline.execute").max_response_bytes == LIMIT
    assert operations.get_operation("exa.search").max_response_bytes == LIMIT
    assert operations.get_operation("openrouter.chat").max_response_bytes == 1_048_576
