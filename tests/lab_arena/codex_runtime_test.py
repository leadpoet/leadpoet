"""Real HTTP/Unix broker boundary and optional native Codex tool-loop proof."""

from __future__ import annotations

import json
import os
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request

import httpx
import pytest

from lab_arena import broker as br, operations as ops, runner, runtime, shim, web_egress
from lab_arena import lab_arena_codex as codex
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, FakeLedgerStore, make_broker, price_table
from tests.lab_arena.test_lab_arena_runner import lease


NATIVE_MODELS = (
    "openai/gpt-6-astra", "openai/gpt-5.6-sol", "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna", "openai/gpt-5.5",
)


def response(output=None, **changes):
    return dict({
        "id": "gen-codex-1", "object": "response", "created_at": 1789488000,
        "status": "completed", "model": "openai/gpt-4o-mini", "error": None,
        "output": output or [{"id": "msg-1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "ARENA_CODEX_OK", "annotations": []}]}],
        "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20, "cost": 0.000012},
    }, **changes)


@contextmanager
def broker_socket(monkeypatch, transport=None, store=None, *, priced_models=()):
    broker, store, transport = make_broker(transport=transport, store=store)
    if priced_models:
        # Synthetic test prices admit model selection. Production still uses
        # the fetched, validated OpenRouter price allowlist.
        table = price_table()
        for model in priced_models:
            table["models"][model] = dict(table["models"]["openai/gpt-4o-mini"])
        broker._price_table = br.validate_price_table(table)

    class Api:
        def provider(self, run_id, lease_token, frame):
            return broker.execute(CONTEXT, **frame).to_document()

    with tempfile.TemporaryDirectory(prefix="codex-test-", dir="/tmp") as directory:
        path = Path(directory) / "worker.sock"
        web_path = Path(directory) / runtime.SANDBOX_WEB_SOCKET_NAME
        state = runner.RunState(lease=lease("r1"), lease_token="tok-r1")
        server = runner.WorkerSocketServer(path, Api(), state)
        server.start()
        web_server = web_egress.WebEgressServer(web_path)
        try:
            web_server.start()
            monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", str(path))
            monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", str(web_path))
            yield store, transport, path
        finally:
            try:
                web_server.stop()
            finally:
                server.stop()


def test_broker_socket_starts_required_web_egress(monkeypatch):
    with broker_socket(monkeypatch) as (_store, _transport, worker_path):
        web_path = Path(os.environ["LAB_ARENA_WEB_EGRESS_SOCKET"])
        assert worker_path.is_socket()
        assert web_path == worker_path.parent / runtime.SANDBOX_WEB_SOCKET_NAME
        assert web_path.is_socket()
    assert not web_path.exists()


@pytest.mark.parametrize("extra", [
    {"stream": True}, {"store": True}, {"previous_response_id": "resp-other"},
    {"provider": {"allow_fallbacks": True}}, {"background": True},
    {"tools": [{"type": "web_search"}]}, {"tools": [{"type": "mcp", "server_url": "https://example.com"}]},
    {"input": [{"type": "item_reference", "id": "remote-item"}]},
    {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]}]},
    {"max_output_tokens": 32769}, {"input": True}, {"tools": {}},
])
def test_closed_responses_operation_rejects_unpriced_or_remote_work(extra):
    with pytest.raises(ops.OperationRequestError):
        ops.validate_operation_request("openrouter.responses", {"model": "openai/gpt-4o-mini", "input": "hi", **extra})


def test_responses_native_tool_history_and_caps():
    params = ops.validate_operation_request("openrouter.responses", {
        "model": "openai/gpt-4o-mini",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "research"}]},
            {"type": "function_call", "id": "fc-1", "call_id": "call-1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "tool evidence"},
            {"type": "reasoning", "id": "rs-1", "summary": [], "encrypted_content": "opaque"},
        ],
        "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}],
    })
    normalized = br.normalized_request("openrouter.responses", params)
    assert normalized["max_output_tokens"] == 4096
    assert "max_tokens" not in normalized
    outbound = ops.build_outbound_request("openrouter.responses", normalized)
    assert outbound.url == "https://openrouter.ai/api/v1/responses"
    body = json.loads(outbound.body)
    assert body["stream"] is False and body["store"] is False
    assert body["provider"] == dict(ops.OPENROUTER_STRICT_PROVIDER_POLICY)


def native_wire_request(model, style, *, continued=False):
    """Small, path-free forms taken from Codex 0.154.0's recorded wire shapes."""
    function = {"type": "function", "name": "arena_inspect", "description": "Inspect local fixture", "strict": True,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}
    custom = {"type": "custom", "name": "exec", "description": "Call a local MCP tool", "format": {"type": "text"}}
    namespace = {"type": "namespace", "name": "functions", "description": "Local code mode", "tools": [custom]}
    items = [{"role": "user", "content": [{"type": "input_text", "text": "Inspect the local fixture."}]}]
    tools = [function] if style == "flat" else []
    if style == "lite":
        items.insert(0, {"type": "additional_tools", "id": "at-local", "role": "developer", "tools": [namespace]})
    if continued:
        if style == "flat":
            items.extend([
                {"type": "function_call", "call_id": "call-1", "name": "arena_inspect",
                 "arguments": "{}", "status": "completed"},
                {"type": "function_call_output", "call_id": "call-1", "output": "ARENA_MCP_FIRST_OK"},
            ])
        else:
            items.extend([
                {"type": "custom_tool_call", "call_id": "call-1", "name": "exec", "namespace": "functions",
                 "input": "text(await tools.mcp__fixture__arena_inspect({}));", "status": "completed"},
                {"type": "custom_tool_call_output", "call_id": "call-1",
                 "output": [{"type": "input_text", "text": "ARENA_MCP_FIRST_OK"}]},
            ])
        items.extend([
            {"role": "user", "content": [{"type": "input_text", "text": "Compacted summary: first local MCP result was ARENA_MCP_FIRST_OK; inspect again."}]},
        ])
    reasoning = {"effort": "high", "context": "all_turns"} if style == "lite" else {"effort": "high"}
    return {"model": model, "input": items, "tools": tools, "reasoning": reasoning,
            "include": ["reasoning.encrypted_content"], "parallel_tool_calls": False,
            "prompt_cache_key": "local-fixture", "text": {"verbosity": "low"}, "tool_choice": "auto",
            "client_metadata": {"source": "codex-cli-0.154.0"}, "stream": True, "store": False}


@pytest.mark.parametrize("model,style", [
    ("openai/gpt-5.5", "flat"), ("openai/gpt-5.6-luna", "lite"),
])
def test_recorded_native_wire_crosses_bridge_worker_broker_and_bills(monkeypatch, model, style):
    """The bridge, worker frame, operation and ledger all see native fields."""
    first_output = ({"type": "function_call", "id": "fc-1", "call_id": "call-1", "name": "arena_inspect",
                     "arguments": "{}", "status": "completed"} if style == "flat" else
                    {"type": "custom_tool_call", "id": "ct-1", "call_id": "call-1", "name": "exec",
                     "namespace": "functions", "input": "text(await tools.mcp__fixture__arena_inspect({}));", "status": "completed"})
    outputs = [[first_output],
               [{"id": "msg-2", "type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "ARENA_NATIVE_WIRE_OK", "annotations": []}]}]]
    transport = FakeTransport([(200, response(model=model, output=output, id="resp-%d" % index))
                               for index, output in enumerate(outputs, 1)])
    with broker_socket(monkeypatch, transport, priced_models=(model,)) as (store, transport, path), codex.ResponsesBridge(str(path)) as bridge:
        with httpx.Client(trust_env=False) as client:
            replies = [client.post(bridge.base_url + "/responses",
                                   headers={"Authorization": "Bearer " + bridge.token},
                                   json=native_wire_request(model, style, continued=index == 1))
                       for index in range(2)]
    assert [reply.status_code for reply in replies] == [200, 200], [reply.text for reply in replies]
    assert "event: response.completed" in replies[0].text
    assert "ARENA_NATIVE_WIRE_OK" in replies[1].text
    assert len(transport.sent) == 2
    outbound = [json.loads(sent["body"]) for sent in transport.sent]
    assert all(sent["url"] == "https://openrouter.ai/api/v1/responses" for sent in transport.sent)
    assert all(body["model"] == model for body in outbound)
    if style == "lite":
        assert all(body["reasoning"]["context"] == "all_turns" for body in outbound)
    assert all(body["max_output_tokens"] == 16384 for body in outbound)
    if model == br.OPENROUTER_LUNA_RESPONSES_MODEL:
        assert all(
            body["provider"]["only"] == ["azure/us", "azure/eu"]
            and "order" not in body["provider"]
            and "require_parameters" not in body["provider"]
            and body["provider"]["allow_fallbacks"] is True
            and body["provider"]["data_collection"] == "deny"
            and body["provider"]["zdr"] is True
            for body in outbound
        )
    else:
        assert all(
            body["provider"] == dict(ops.OPENROUTER_STRICT_PROVIDER_POLICY)
            for body in outbound
        )
    assert "client_metadata" not in outbound[0]
    if style == "lite":
        assert outbound[0]["input"][0]["type"] == "additional_tools"
        assert outbound[1]["input"][-2]["output"] == [{"type": "input_text", "text": "ARENA_MCP_FIRST_OK"}]
    else:
        assert outbound[0]["tools"][0]["type"] == "function"
        assert outbound[1]["input"][-2]["output"] == "ARENA_MCP_FIRST_OK"
    assert "Compacted summary" in outbound[1]["input"][-1]["content"][0]["text"]
    assert len(store.calls) == 2
    assert all(call["kind"] == "settlement" and call["terminal"]["call_succeeded"] is True
               for call in store.calls.values())
    assert sum(call["actual"] for call in store.calls.values()) == 24


def test_bridge_chunks_recorded_large_custom_tool_result_without_text_loss(monkeypatch):
    """The real Codex 0.154 typed output shape crosses the frozen operation."""
    model = "openai/gpt-5.6-luna"
    request = native_wire_request(model, "lite", continued=True)
    original = "r" * 32_000 + "🙂" * 8_870
    request["input"][-2]["output"] = [{"type": "input_text", "text": original}]
    transport = FakeTransport([(200, response(model=model))])

    with broker_socket(monkeypatch, transport, priced_models=(model,)) as (store, transport, path), codex.ResponsesBridge(str(path)) as bridge:
        with httpx.Client(trust_env=False) as client:
            reply = client.post(
                bridge.base_url + "/responses",
                headers={"Authorization": "Bearer " + bridge.token},
                json=request,
            )

    assert reply.status_code == 200, reply.text
    outbound = json.loads(transport.sent[0]["body"])
    parts = outbound["input"][-2]["output"]
    assert [len(part["text"]) for part in parts] == [32_000, 8_870]
    assert "".join(part["text"] for part in parts) == original
    assert len(store.calls) == 1


def test_bridge_bills_real_worker_and_replays_sse(monkeypatch):
    with broker_socket(monkeypatch, FakeTransport([(200, response())])) as (store, transport, path), codex.ResponsesBridge(str(path)) as bridge:
        with httpx.Client(trust_env=False) as client:
            reply = client.post(bridge.base_url + "/responses", headers={"Authorization": "Bearer " + bridge.token}, json={"model": "openai/gpt-4o-mini", "input": "hi", "stream": True, "store": False})
        assert reply.status_code == 200, reply.text
        assert "event: response.completed" in reply.text
        assert "ARENA_CODEX_OK" in reply.text
        assert len(transport.sent) == 1
        assert next(iter(store.calls.values()))["actual"] == 12
        assert next(iter(store.calls.values()))["terminal"]["call_succeeded"] is True


def test_bridge_auth_and_stateful_requests_never_dispatch(monkeypatch):
    with broker_socket(monkeypatch) as (store, transport, path), codex.ResponsesBridge(str(path)) as bridge:
        with httpx.Client(trust_env=False) as client:
            assert client.post(bridge.base_url + "/responses", json={}).status_code == 401
            stateful = client.post(bridge.base_url + "/responses", headers={"Authorization": "Bearer " + bridge.token}, json={"store": True})
            assert stateful.status_code == 400
            assert stateful.json() == {"error": {
                "code": "stateful_request",
                "message": "invalid Responses request or reply",
            }}
            assert client.post(bridge.base_url + "/chat/completions", json={}).status_code == 404
        assert not store.calls and not transport.sent


@pytest.mark.parametrize(("payload", "code", "web_search"), [
    (b"not-json", "invalid_json", "disabled"),
    ([], "invalid_request_schema", "disabled"),
    ({"stream": "true"}, "invalid_stream", "disabled"),
    ({"max_output_tokens": 0}, "invalid_output_token_limit", "disabled"),
    ({"tools": {"bad": "shape"}}, "invalid_tools", "disabled"),
    ({"tools": [{"type": "openrouter:web_search"}]},
     "hosted_tools_forbidden", "disabled"),
    ({"tools": [{"type": "web_search"}]},
     "web_search_unavailable", "disabled"),
    ({"tools": [{"type": "web_search", "unknown": True}]},
     "invalid_web_search_tool", "live"),
    ({"tools": [{"type": "web_search", "external_web_access": False}]},
     "live_web_search_required", "live"),
    ({"tools": [{"type": "web_search", "search_context_size": "all"}]},
     "invalid_web_search_context", "live"),
])
def test_bridge_malformed_requests_return_closed_codes_before_dispatch(
    monkeypatch, payload, code, web_search,
):
    monkeypatch.setattr(
        codex,
        "_dispatch",
        lambda *_args, **_kwargs: pytest.fail("invalid request was dispatched"),
    )
    with codex.ResponsesBridge(
        "/synthetic-unused.sock", web_search=web_search,
    ) as bridge, httpx.Client(trust_env=False) as client:
        request = {"content": payload} if isinstance(payload, bytes) else {"json": payload}
        reply = client.post(
            bridge.base_url + "/responses",
            headers={"Authorization": "Bearer " + bridge.token},
            **request,
        )

    assert reply.status_code == 400
    assert reply.json() == {"error": {
        "code": code,
        "message": "invalid Responses request or reply",
    }}


def test_bridge_oversize_request_returns_closed_code_before_dispatch(monkeypatch):
    monkeypatch.setattr(
        codex,
        "_dispatch",
        lambda *_args, **_kwargs: pytest.fail("oversize request was dispatched"),
    )
    with codex.ResponsesBridge("/synthetic-unused.sock") as bridge:
        with socket.create_connection(
            ("127.0.0.1", bridge.server.server_port), timeout=2,
        ) as connection:
            connection.sendall(
                b"POST /v1/responses HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                + ("Authorization: Bearer " + bridge.token + "\r\n").encode()
                + ("Content-Length: " + str(codex.MAX_REQUEST_BYTES + 1) + "\r\n").encode()
                + b"Connection: close\r\n\r\n"
            )
            chunks = []
            while True:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)

    head, body = b"".join(chunks).split(b"\r\n\r\n", 1)
    assert b" 400 " in head
    assert json.loads(body) == {"error": {
        "code": "request_body_too_large",
        "message": "invalid Responses request or reply",
    }}


def test_bridge_valid_request_keeps_existing_response_behavior(monkeypatch):
    requests = []

    def dispatch(_socket_path, body, **_kwargs):
        requests.append(body)
        return 200, b'{"status":"completed"}'

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    with codex.ResponsesBridge(
        "/synthetic-unused.sock",
    ) as bridge, httpx.Client(trust_env=False) as client:
        reply = client.post(
            bridge.base_url + "/responses",
            headers={"Authorization": "Bearer " + bridge.token},
            json={"model": "openai/gpt-5.6-luna", "input": "synthetic"},
        )

    assert reply.status_code == 200
    assert reply.content == b'{"status":"completed"}'
    assert requests == [{
        "model": "openai/gpt-5.6-luna",
        "input": "synthetic",
        "max_output_tokens": codex.DEFAULT_MAX_OUTPUT_TOKENS,
    }]


@pytest.mark.parametrize("exception_type", [ValueError, TypeError, KeyError])
def test_bridge_unexpected_exception_keeps_generic_error_without_secret(
    monkeypatch, exception_type,
):
    secret = "private-upstream-frame-do-not-return"

    def dispatch(*_args, **_kwargs):
        raise exception_type(secret)

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    with codex.ResponsesBridge(
        "/synthetic-unused.sock",
    ) as bridge, httpx.Client(trust_env=False) as client:
        reply = client.post(
            bridge.base_url + "/responses",
            headers={"Authorization": "Bearer " + bridge.token},
            json={"model": "openai/gpt-5.6-luna", "input": "synthetic"},
        )

    assert reply.status_code == 400
    assert reply.content == codex._INVALID_RESPONSES_ERROR
    assert reply.json() == {
        "error": {"message": "invalid Responses request or reply"},
    }
    assert secret.encode() not in reply.content


def test_bridge_error_code_serialization_is_closed_and_extractable():
    for code in codex._BRIDGE_REQUEST_ERROR_CODES:
        document = json.loads(codex._invalid_responses_error(code))
        assert document == {"error": {
            "code": code,
            "message": "invalid Responses request or reply",
        }}

    secret = "untrusted-error-code-secret"
    assert codex._invalid_responses_error(secret) == codex._INVALID_RESPONSES_ERROR
    assert secret.encode() not in codex._invalid_responses_error(secret)
    with pytest.raises(ValueError, match="invalid bridge request error code"):
        codex._BridgeRequestError(secret)


def test_live_web_search_is_injected_as_one_bounded_native_server_tool(monkeypatch):
    payload = response(
        id="gen-live-search",
        output=[
            {
                "id": "ws-1", "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "current example"},
            },
            {
                "id": "msg-search", "type": "message", "role": "assistant",
                "status": "completed", "content": [{
                    "type": "output_text", "text": "Current result",
                    "annotations": [{
                        "type": "url_citation", "url": "https://example.com/source",
                        "title": "Example source", "start_index": 0,
                        "end_index": 14,
                    }],
                }],
            },
        ],
        usage={
            "input_tokens": 10, "output_tokens": 10, "total_tokens": 20,
            "cost": 0.000012,
            "server_tool_use": {"web_search_requests": 1},
        },
    )
    posts = []

    def dispatch(_socket_path, parameters, **_kwargs):
        posts.append(parameters)
        return 200, json.dumps(payload, separators=(",", ":")).encode()

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    with codex.ResponsesBridge(
            "/fixture-worker.sock", web_search="live") as bridge:
        with httpx.Client(trust_env=False) as client:
            result = client.post(
                bridge.base_url + "/responses",
                headers={"Authorization": "Bearer " + bridge.token},
                json={
                    "model": "openai/gpt-4o-mini", "input": "Search now",
                    "tools": None, "stream": True,
                },
            )

    assert result.status_code == 200
    assert len(posts) == 1
    sent = posts[0]
    assert sent["tools"] == [{
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "native",
            "max_uses": codex.WEB_SEARCH_MAX_TOOL_CALLS,
            "max_total_results": codex.WEB_SEARCH_MAX_TOTAL_RESULTS,
        },
    }]
    assert sent["max_tool_calls"] == codex.WEB_SEARCH_MAX_TOOL_CALLS
    assert codex.WEB_SEARCH_MAX_TOOL_CALLS == ops.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS
    assert codex.WEB_SEARCH_MAX_TOTAL_RESULTS == ops.OPENROUTER_WEB_SEARCH_MAX_TOTAL_RESULTS
    assert b'"type":"web_search_call"' in result.content
    assert b'"type":"url_citation"' in result.content


def test_standalone_page_fetch_forces_loopback_proxy_and_extracts_visible_text(
    monkeypatch,
):
    private_hits = []
    proxy_hits = []

    class PrivateHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            private_hits.append(self.path)
            body = b"PRIVATE_LOOPBACK_CONTENT"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class ProxyHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            proxy_hits.append(self.path)
            if self.path.endswith("/private"):
                self.send_response(302)
                self.send_header("Location", "http://redirect.example/final")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = (
                b"<html><head><style>HIDDEN_STYLE</style></head>"
                b"<body><h1>Fixture heading</h1><script>HIDDEN_SCRIPT</script>"
                b"<p>Visible fixture text.</p></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    private = ThreadingHTTPServer(("127.0.0.1", 0), PrivateHandler)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (private, proxy)
    ]
    for thread in threads:
        thread.start()
    try:
        target = "http://127.0.0.1:%d/private" % private.server_port
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setenv(
            "LAB_ARENA_WEB_PROXY_URL",
            "http://127.0.0.1:%d" % proxy.server_port,
        )
        final_url, lines = codex._fetch_visible_page(
            target, response_deadline=time.monotonic() + 5,
        )
    finally:
        for server in (private, proxy):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(2)

    assert final_url == "http://redirect.example/final"
    assert proxy_hits == [target, "http://redirect.example/final"]
    assert private_hits == []
    assert lines == ["Fixture heading", "Visible fixture text."]


def test_standalone_https_request_is_forced_through_loopback_proxy():
    request = Request("https://public.example/source")
    handler = codex._LoopbackProxyHandler({
        "https": "http://127.0.0.1:8123",
    })

    assert handler.proxy_open(
        request, "http://127.0.0.1:8123", "https",
    ) is None
    assert request.host == "127.0.0.1:8123"
    assert request._tunnel_host == "public.example"


def test_standalone_open_and_find_return_numbered_raw_visible_text(monkeypatch):
    fetches = []

    def fetch(url, *, response_deadline):
        fetches.append((url, response_deadline))
        return "https://public.example/final", [
            "Heading", "First evidence line", "Needle evidence", "Last line",
        ]

    monkeypatch.setattr(codex, "_fetch_visible_page", fetch)
    deadline = time.monotonic() + 5
    opened = json.loads(codex._standalone_page_output({
        "commands": {"open": [{
            "ref_id": "https://public.example/page", "lineno": 0,
        }]},
        "settings": {"external_web_access": True},
    }, response_deadline=deadline))
    found = json.loads(codex._standalone_page_output({
        "commands": {"find": [{
            "ref_id": "https://public.example/page", "pattern": "needle",
        }]},
        "settings": {"external_web_access": True},
    }, response_deadline=deadline))

    assert "Open https://public.example/final" in opened["output"]
    assert "L1: Heading" in opened["output"]
    assert "Find 'needle' in https://public.example/final" in found["output"]
    assert "L3: Needle evidence" in found["output"]
    assert len(fetches) == 2


def test_standalone_open_uses_shared_gate_and_guard(monkeypatch):
    calls = []

    class Gate:
        def acquire(self, *, timeout):
            calls.append(("acquire", timeout))
            return True

        def release(self):
            calls.append(("release", None))

    def guard():
        calls.append(("guard", None))
        return True

    monkeypatch.setattr(
        codex, "_fetch_visible_page",
        lambda url, **_kwargs: (url, ["Visible source text"]),
    )
    with codex.ResponsesBridge(
        "/fixture-worker.sock", web_search="live",
        request_gate=Gate(), request_guard=guard,
    ) as bridge, httpx.Client(trust_env=False) as client:
        result = client.post(
            bridge.base_url + "/alpha/search",
            headers={"Authorization": "Bearer " + bridge.token},
            json={
                "commands": {"open": [{
                    "ref_id": "https://public.example/source",
                }]},
                "settings": {"external_web_access": True},
            },
        )

    assert result.status_code == 200
    assert "Visible source text" in result.json()["output"]
    assert [name for name, _value in calls] == ["acquire", "guard", "release"]


@pytest.mark.parametrize("commands", [
    {
        "search_query": [{"q": "fixture"}],
        "open": [{"ref_id": "https://public.example/page"}],
    },
    {
        "open": [
            {"ref_id": "https://public.example/valid"},
            {"ref_id": "opaque-search-reference"},
        ],
    },
])
def test_standalone_mixed_or_invalid_page_commands_do_not_fetch(
    monkeypatch, commands,
):
    monkeypatch.setattr(
        codex, "_fetch_visible_page",
        lambda *_args, **_kwargs: pytest.fail("invalid command fetched a page"),
    )
    result = json.loads(codex._standalone_page_output({
        "commands": commands,
        "settings": {"external_web_access": True},
    }, response_deadline=time.monotonic() + 5))

    assert "No source evidence was retrieved" in result["output"]


def test_standalone_search_tool_miss_continues_without_source_claim(monkeypatch):
    sent = []

    def dispatch(_socket_path, body, **_kwargs):
        sent.append(body)
        payload = response(output=[
            {
                "id": "ws-miss", "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "fixture"},
            },
            {
                "id": "msg-miss", "type": "message", "role": "assistant",
                "status": "completed", "content": [{
                    "type": "output_text", "text": "Tool error", "annotations": [],
                }],
            },
        ])
        return 200, json.dumps(payload).encode()

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    with codex.ResponsesBridge(
        "/fixture-worker.sock", web_search="live",
    ) as bridge, httpx.Client(trust_env=False) as client:
        result = client.post(
            bridge.base_url + "/alpha/search",
            headers={"Authorization": "Bearer " + bridge.token},
            json={
                "model": "openai/gpt-5.6-luna",
                "commands": {"search_query": [{"q": "fixture"}]},
                "settings": {"external_web_access": True},
                "max_output_tokens": 10_000,
            },
        )

    assert result.status_code == 200
    assert "no cited source results" in result.json()["output"]
    assert len(sent) == 1
    assert sent[0]["tools"] == [{
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "exa",
            "max_uses": codex.WEB_SEARCH_MAX_TOOL_CALLS,
            "max_total_results": codex.WEB_SEARCH_MAX_TOTAL_RESULTS,
        },
    }]
    assert sent[0]["tool_choice"] == "auto"
    assert sent[0]["max_output_tokens"] == 4096


def test_standalone_exa_search_returns_only_structured_citations():
    document = response(output=[
        {
            "id": "st_tmp_fixture", "type": "openrouter:web_search",
            "status": "completed", "action": {
                "type": "search", "query": "fixture company series a",
                "sources": [
                    {"type": "url", "url": "https://example.com/series-a"},
                    {"type": "url", "url": "https://news.example/fixture"},
                ],
            },
        },
        {
            "id": "msg-exa", "type": "message", "role": "assistant",
            "status": "completed", "content": [{
                "type": "output_text",
                "text": "Fixture raised a Series A.",
                "annotations": [{
                    "type": "url_citation",
                    "url": "https://example.com/series-a",
                    "title": "Fixture Series A", "start_index": 0,
                    "end_index": 27,
                }],
            }],
        },
    ])

    result = json.loads(codex._standalone_search_output(document))["output"]

    assert "Fixture raised a Series A." in result
    assert "https://example.com/series-a" in result
    assert "open each source URL" in result


@pytest.mark.parametrize("document", [
    response(output=[
        {
            "id": "st_tmp_empty", "type": "openrouter:web_search",
            "status": "completed", "action": {
                "type": "search", "query": "fixture", "sources": [],
            },
        },
        {
            "id": "msg-empty", "type": "message", "role": "assistant",
            "status": "completed", "content": [{
                "type": "output_text", "text": "https://example.com/spoof",
                "annotations": [],
            }],
        },
    ]),
    response(output=[
        {
            "id": "st_tmp_plain", "type": "openrouter:web_search",
            "status": "completed", "action": {
                "type": "search", "query": "fixture", "sources": [{
                    "type": "url", "url": "https://example.com/source",
                }],
            },
        },
        {
            "id": "msg-plain", "type": "message", "role": "assistant",
            "status": "completed", "content": [{
                "type": "output_text", "text": "https://example.com/spoof",
                "annotations": [],
            }],
        },
    ]),
])
def test_standalone_exa_miss_or_plain_url_is_not_source_evidence(document):
    result = json.loads(codex._standalone_search_output(document))["output"]

    assert "No source evidence was retrieved" in result
    assert "https://example.com/spoof" not in result


@pytest.mark.parametrize("document", [
    [],
    {"status": "completed", "output": ["not-an-item"]},
    {
        "status": "completed",
        "output": [
            {"type": "web_search_call", "status": "completed"},
            {"type": "message", "content": ["not-a-part"]},
        ],
    },
    {
        "status": "completed",
        "output": [
            {"type": "web_search_call", "status": "completed"},
            {"type": "message", "content": [{
                "type": "output_text", "text": 3,
                "annotations": ["not-an-annotation"],
            }]},
        ],
    },
    {
        "status": "completed",
        "output": [{
            "type": "openrouter:web_search", "status": "completed",
            "action": "not-an-action",
        }],
    },
])
def test_standalone_malformed_provider_output_becomes_local_tool_miss(document):
    result = json.loads(codex._standalone_search_output(document))

    assert "No source evidence was retrieved" in result["output"]


def test_standalone_search_invalid_token_limit_never_dispatches(monkeypatch):
    monkeypatch.setattr(
        codex, "_dispatch",
        lambda *_args, **_kwargs: pytest.fail("invalid search was dispatched"),
    )
    with codex.ResponsesBridge(
        "/fixture-worker.sock", web_search="live",
    ) as bridge, httpx.Client(trust_env=False) as client:
        result = client.post(
            bridge.base_url + "/alpha/search",
            headers={"Authorization": "Bearer " + bridge.token},
            json={
                "model": "openai/gpt-5.6-luna",
                "commands": {"search_query": [{"q": "fixture"}]},
                "max_output_tokens": 0,
            },
        )

    assert result.status_code == 200
    assert "No source evidence was retrieved" in result.json()["output"]


@pytest.mark.parametrize("commands", [
    {"search_query": [{"q": "   "}]},
    {"click": [{"ref_id": "https://example.com", "id": 1}]},
])
def test_standalone_unsupported_command_is_a_nonfatal_tool_miss(monkeypatch, commands):
    monkeypatch.setattr(
        codex, "_dispatch",
        lambda *_args, **_kwargs: pytest.fail("invalid search was dispatched"),
    )
    with codex.ResponsesBridge(
        "/fixture-worker.sock", web_search="live",
    ) as bridge, httpx.Client(trust_env=False) as client:
        result = client.post(
            bridge.base_url + "/alpha/search",
            headers={"Authorization": "Bearer " + bridge.token},
            json={"model": "openai/gpt-5.6-luna", "commands": commands},
        )
    assert result.status_code == 200
    assert "No source evidence was retrieved" in result.json()["output"]


def test_parallel_standalone_search_calls_queue_within_the_session_deadline(
    monkeypatch,
):
    first_entered = threading.Event()
    release_first = threading.Event()
    dispatched = []

    def dispatch(_socket_path, body, **_kwargs):
        dispatched.append(body)
        if len(dispatched) == 1:
            first_entered.set()
            assert release_first.wait(2)
        return 200, json.dumps(response(output=[
            {
                "id": "ws-empty", "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "fixture"},
            },
        ])).encode()

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    statuses = []
    with codex.ResponsesBridge(
        "/fixture-worker.sock", web_search="live",
        response_deadline=time.monotonic() + 5,
    ) as bridge:
        def search(query):
            with httpx.Client(trust_env=False) as client:
                result = client.post(
                    bridge.base_url + "/alpha/search",
                    headers={"Authorization": "Bearer " + bridge.token},
                    json={
                        "model": "openai/gpt-5.6-luna",
                        "commands": {"search_query": [{"q": query}]},
                    },
                )
                statuses.append(result.status_code)

        first = threading.Thread(target=search, args=("first",))
        second = threading.Thread(target=search, args=("second",))
        first.start()
        assert first_entered.wait(2)
        second.start()
        time.sleep(0.05)
        assert second.is_alive()
        release_first.set()
        first.join(2)
        second.join(2)

    assert sorted(statuses) == [200, 200]
    assert len(dispatched) == 2


@pytest.mark.parametrize("mode", ["disabled", "live"])
def test_caller_cannot_supply_an_openrouter_hosted_tool(monkeypatch, mode):
    monkeypatch.setattr(
        codex, "_dispatch",
        lambda *_args: pytest.fail("caller-supplied hosted tool was dispatched"),
    )
    with codex.ResponsesBridge(
            "/fixture-worker.sock", web_search=mode) as bridge:
        with httpx.Client(trust_env=False) as client:
            result = client.post(
                bridge.base_url + "/responses",
                headers={"Authorization": "Bearer " + bridge.token},
                json={
                    "model": "openai/gpt-5.6-luna", "input": "Search",
                    "tools": [{"type": "openrouter:web_search"}],
                },
            )
    assert result.status_code == 400


@pytest.mark.skipif(
    not os.environ.get("ARENA_TEST_CODEX_BINARY"),
    reason="set ARENA_TEST_CODEX_BINARY for the real CLI web-search proof",
)
def test_real_codex_live_web_search_crosses_bridge_with_history_and_citation(
    monkeypatch, tmp_path,
):
    _pin_real_codex(monkeypatch)
    posts = []

    def dispatch(_socket_path, parameters, **_kwargs):
        ops.validate_operation_request("openrouter.responses", parameters)
        posts.append(parameters)
        if len(posts) == 1:
            assert any(
                item.get("type") == "additional_tools"
                for item in parameters["input"]
            )
            document = response(output=[
                {
                    "id": "ws-real", "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "current fixture"},
                },
                {
                    "id": "msg-search", "type": "message", "role": "assistant",
                    "phase": "commentary", "status": "completed", "content": [{
                        "type": "output_text", "text": "Found a source.",
                        "annotations": [{
                            "type": "url_citation",
                            "url": "https://example.com/source",
                            "title": "Example source", "start_index": 8,
                            "end_index": 14,
                        }],
                    }],
                },
                {
                    "id": "ct-search", "type": "custom_tool_call",
                    "call_id": "call-search", "name": "exec",
                    "namespace": "functions",
                    "input": 'text("ARENA_LOCAL_OK")', "status": "completed",
                },
            ])
        else:
            assert any(
                item.get("type") == "web_search_call"
                and item.get("action", {}).get("query") == "current fixture"
                for item in parameters["input"]
            )
            assert any(
                item.get("type") == "custom_tool_call_output"
                and "ARENA_LOCAL_OK" in json.dumps(item.get("output"))
                for item in parameters["input"]
            )
            document = response(
                id="gen-search-final",
                output=[{
                    "id": "msg-real", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{
                        "type": "output_text",
                        "text": "ARENA_WEB_SEARCH_OK https://example.com/source",
                        "annotations": [{
                            "type": "url_citation",
                            "url": "https://example.com/source",
                            "title": "Example source", "start_index": 20,
                            "end_index": 46,
                        }],
                    }],
                }],
            )
        return 200, json.dumps(document, separators=(",", ":")).encode()

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    result = codex.run(
        "Search the web and report the fixture result.",
        model="openai/gpt-5.6-luna", cwd=tmp_path,
        timeout_seconds=60, web_search="live",
    )

    assert "ARENA_WEB_SEARCH_OK" in result
    assert len(posts) == 2
    for post in posts:
        assert post["tools"][-1]["type"] == "openrouter:web_search"
        assert post["tools"][-1]["parameters"]["engine"] == "native"
        assert post["max_tool_calls"] == codex.WEB_SEARCH_MAX_TOOL_CALLS


@pytest.mark.skipif(
    not os.environ.get("ARENA_TEST_CODEX_BINARY"),
    reason="set ARENA_TEST_CODEX_BINARY for the real standalone-search proof",
)
def test_real_codex_standalone_web_search_crosses_accounted_bridge(
    monkeypatch, tmp_path,
):
    _pin_real_codex(monkeypatch)
    fetched_pages = []

    def fetch_page(url, *, response_deadline):
        fetched_pages.append((url, response_deadline))
        return url, [
            "Standalone source heading",
            "EXACT_STANDALONE_OPEN_RESULT from raw visible page text.",
        ]

    monkeypatch.setattr(codex, "_fetch_visible_page", fetch_page)

    class StandaloneSearchTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.requests = []

        def send(self, **kwargs):
            body = json.loads(kwargs["body"])
            self.requests.append(body)
            search_adapter = (
                body.get("tool_choice") == "auto"
                and isinstance(body.get("input"), str)
                and "Execute the supplied web search" in body.get("instructions", "")
            )
            if len(self.requests) == 1:
                assert any(
                    item.get("type") == "additional_tools"
                    and "web" in json.dumps(item.get("tools"))
                    for item in body["input"]
                )
                output = [{
                    "type": "custom_tool_call", "id": "ct-search",
                    "call_id": "call-standalone-search", "name": "exec",
                    "namespace": "functions", "status": "completed",
                    "input": (
                        "const missed = await tools.web__run({"
                        "search_query: [{q: '   '}]"
                        "});"
                        "const searched = await tools.web__run({"
                        "search_query: [{q: 'standalone executor proof'}]"
                        "});"
                        "const opened = await tools.web__run({"
                        "open: [{ref_id: "
                        "'https://example.com/standalone-source'}]"
                        "});"
                        "text(JSON.stringify({missed, searched, opened}));"
                    ),
                }]
            elif search_adapter:
                assert body["tools"] == [{
                    "type": "openrouter:web_search",
                    "parameters": {
                        "engine": "exa",
                        "max_uses": codex.WEB_SEARCH_MAX_TOOL_CALLS,
                        "max_total_results": codex.WEB_SEARCH_MAX_TOTAL_RESULTS,
                    },
                }]
                assert body["max_tool_calls"] == codex.WEB_SEARCH_MAX_TOOL_CALLS
                assert "standalone executor proof" in body["input"]
                output = [
                    {
                        "id": "ws-standalone", "type": "openrouter:web_search",
                        "status": "completed", "action": {
                            "type": "search",
                            "query": "standalone executor proof",
                            "sources": [{
                                "type": "url",
                                "url": "https://example.com/standalone-source",
                            }],
                        },
                    },
                    {
                        "id": "msg-standalone", "type": "message",
                        "role": "assistant", "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": "EXACT_STANDALONE_SEARCH_RESULT",
                            "annotations": [{
                                "type": "url_citation",
                                "url": "https://example.com/standalone-source",
                                "title": "Standalone source",
                                "start_index": 0,
                                "end_index": 30,
                            }],
                        }],
                    },
                ]
            else:
                assert any(
                    item.get("type") == "custom_tool_call_output"
                    and "No source evidence was retrieved"
                    in json.dumps(item.get("output"))
                    and "EXACT_STANDALONE_SEARCH_RESULT"
                    in json.dumps(item.get("output"))
                    and "EXACT_STANDALONE_OPEN_RESULT"
                    in json.dumps(item.get("output"))
                    and "https://example.com/standalone-source"
                    in json.dumps(item.get("output"))
                    for item in body["input"]
                )
                output = [{
                    "id": "msg-final", "type": "message",
                    "role": "assistant", "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "ARENA_STANDALONE_SEARCH_OK",
                        "annotations": [],
                    }],
                }]
            self.responses.append((200, response(
                model=body["model"], id="standalone-%d" % len(self.requests),
                output=output,
            )))
            return super().send(**kwargs)

    transport = StandaloneSearchTransport()
    with broker_socket(
        monkeypatch, transport,
        priced_models=("openai/gpt-5.6-luna",),
    ) as (store, transport, _path):
        result = codex.run(
            "Use tools.web__run from code mode, then report its exact result.",
            model="openai/gpt-5.6-luna", cwd=tmp_path,
            timeout_seconds=60, web_search="live",
        )

    assert "ARENA_STANDALONE_SEARCH_OK" in result
    assert len(transport.requests) == len(store.calls) == 3
    assert sum(
        isinstance(body.get("input"), str)
        and "Execute the supplied web search" in body.get("instructions", "")
        for body in transport.requests
    ) == 1
    assert len(fetched_pages) == 1
    assert fetched_pages[0][0] == "https://example.com/standalone-source"
    assert all(call["kind"] == "settlement" for call in store.calls.values())


def test_responses_budget_refusal_and_incomplete_billing(monkeypatch):
    with broker_socket(monkeypatch, store=FakeLedgerStore(openrouter_capacity=0)) as (store, transport, path):
        status, _ = codex._dispatch(str(path), {"model": "openai/gpt-4o-mini", "input": "hi"})
        assert status >= 400
        assert not transport.sent
    broker, store, _ = make_broker(transport=FakeTransport([(200, response(status="incomplete"))]))
    result = broker.execute(CONTEXT, operation_id="openrouter.responses", parameters={"model": "openai/gpt-4o-mini", "input": "hi"}, action_sequence=1, timeout_ms=120000)
    assert next(iter(store.calls.values()))["terminal"]["call_succeeded"] is False
    assert next(iter(store.calls.values()))["actual"] == 12


@pytest.mark.parametrize("scrapingdog_value", ["lab-arena-brokered-scrapingdog", "test-secret"])
def test_session_isolates_login_and_provider_keys(monkeypatch, scrapingdog_value):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", scrapingdog_value)
    monkeypatch.setenv("CODEX_HOME", "/personal/config")
    with broker_socket(monkeypatch), codex.session(model="openai/gpt-4o-mini") as env:
        assert "OPENROUTER_API_KEY" not in env and "OPENAI_API_KEY" not in env
        if scrapingdog_value == "lab-arena-brokered-scrapingdog":
            assert env["SCRAPINGDOG_API_KEY"] == scrapingdog_value
        else:
            assert "SCRAPINGDOG_API_KEY" not in env
        assert env["CODEX_HOME"] != "/personal/config"
        home = Path(env["CODEX_HOME"])
        assert not (home / "auth.json").exists()
        config = (home / "config.toml").read_text()
        compact_setting = "model_auto_compact_token_limit = 64000"
        assert codex.MODEL_AUTO_COMPACT_TOKEN_LIMIT == 64_000
        assert config.count(compact_setting) == 1
        assert config.index(compact_setting) < config.index("[agents]")
        assert "model_context_window" not in config
        assert 'request_max_retries = 0' in config
        assert 'stream_max_retries = 0' in config
    assert not home.exists()


@pytest.mark.parametrize(
    ("remaining_seconds", "expected_operation_milliseconds", "expected_api_seconds"),
    ((500.0, 433_000, 498.0), (667.0, 600_000, 665.0)),
)
def test_codex_timeout_layers_use_the_original_absolute_response_deadline(
    monkeypatch,
    remaining_seconds,
    expected_operation_milliseconds,
    expected_api_seconds,
):
    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/worker.sock")
    monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", "/egress.sock")
    monkeypatch.setattr(codex.time, "monotonic", lambda: 100.0)
    with codex.session(
        model="openai/gpt-5.6-sol",
        response_deadline=100.0 + remaining_seconds,
    ) as environment:
        config = (Path(environment["CODEX_HOME"]) / "config.toml").read_text()

    sent = bytearray()
    reply = json.dumps({
        "status": 200,
        "body_b64": "e30=",
    }).encode()

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def settimeout(self, timeout):
            assert timeout == remaining_seconds

        def connect(self, _path):
            pass

        def sendall(self, payload):
            sent.extend(payload)

        def recv(self, size):
            framed = len(reply).to_bytes(4, "big") + reply
            offset = getattr(self, "offset", 0)
            chunk = framed[offset:offset + size]
            self.offset = offset + len(chunk)
            return chunk

    monkeypatch.setattr(codex.socket, "socket", lambda *_args: Socket())
    assert codex._dispatch(
        "/worker.sock",
        {"model": "openai/gpt-5.6-sol", "input": "hi"},
        response_deadline=100.0 + remaining_seconds,
    ) == (200, b"{}")
    size = int.from_bytes(sent[:4], "big")
    operation_id, parameters, timeout_ms = shim.decode_operation_frame(sent[4:4 + size])
    assert operation_id == "openrouter.responses"
    assert parameters == {"model": "openai/gpt-5.6-sol", "input": "hi", "max_output_tokens": 4096}
    assert timeout_ms == expected_operation_milliseconds
    assert ops.OPERATIONS["openrouter.responses"].timeout_seconds == 600
    assert runner.MAX_PROVIDER_API_TIMEOUT_SECONDS == 665
    assert codex.PROVIDER_API_HOST_OVERHEAD_MILLISECONDS == 1000 * int(
        ops.BUDGET_ADMISSION_MAX_SECONDS
        + ops.PROVIDER_BILLING_RECONCILIATION_SECONDS
        + ops.PROVIDER_API_TIMEOUT_GRACE_SECONDS
    )

    api_timeouts = []

    class Client:
        def post(self, _url, **kwargs):
            api_timeouts.append(kwargs["timeout"].read)

            class Response:
                status_code = 200

                @staticmethod
                def json():
                    return {"status": "ok"}

            return Response()

    runner.HttpArenaApiClient("http://localhost", client=Client()).provider(
        "run-1",
        "a" * 64,
        {
            "operation_id": "openrouter.responses",
            "parameters": {},
            "timeout_ms": timeout_ms,
            "action_sequence": 0,
        },
    )
    assert api_timeouts == [expected_api_seconds]
    assert expected_api_seconds + (
        codex.BRIDGE_RESPONSE_MARGIN_MILLISECONDS / 1000
    ) <= remaining_seconds

    assert "request_max_retries = 0" in config
    assert "stream_max_retries = 0" in config
    assert "stream_idle_timeout_ms = %d" % int(remaining_seconds * 1000) in config


@pytest.mark.parametrize("remaining_seconds", (67.0, 66.999))
def test_codex_deadline_margin_refuses_before_worker_connect(
    monkeypatch, remaining_seconds,
):
    socket_calls = []
    monkeypatch.setattr(codex.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        codex.socket,
        "socket",
        lambda *_args: socket_calls.append(True),
    )

    with pytest.raises(codex.CodexRuntimeError, match="response deadline"):
        codex._dispatch(
            "/worker.sock",
            {"model": "openai/gpt-5.6-sol", "input": "hi"},
            response_deadline=100.0 + remaining_seconds,
        )

    assert socket_calls == []


def test_codex_client_cancellation_half_closes_and_drains_worker_request(monkeypatch):
    clock = [100.0]
    cancelled = [False]
    reply = json.dumps({"status": 200, "body_b64": "e30="}).encode()
    framed = len(reply).to_bytes(4, "big") + reply
    shutdowns = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def settimeout(self, _timeout):
            return None

        def connect(self, _path):
            return None

        def sendall(self, _payload):
            return None

        def shutdown(self, direction):
            shutdowns.append(direction)

        def recv(self, _size):
            if not cancelled[0]:
                cancelled[0] = True
                raise socket.timeout
            offset = getattr(self, "offset", 0)
            chunk = framed[offset:offset + _size]
            self.offset = offset + len(chunk)
            return chunk

    monkeypatch.setattr(codex.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(codex.socket, "socket", lambda *_args: Socket())

    assert codex._dispatch(
        "/worker.sock",
        {"model": "openai/gpt-5.6-sol", "input": "hi"},
        response_deadline=800.0,
        cancel_requested=lambda: cancelled[0],
    ) == (200, b"{}")
    assert shutdowns == [socket.SHUT_WR]


def test_codex_disconnect_keeps_bridge_active_until_worker_terminal(monkeypatch):
    entered = threading.Event()
    disconnected = threading.Event()
    terminal = threading.Event()

    def dispatch(_socket_path, _document, *, cancel_requested, **_kwargs):
        entered.set()
        for _ in range(100):
            if cancel_requested():
                disconnected.set()
                break
            threading.Event().wait(0.01)
        assert disconnected.is_set()
        assert terminal.wait(1)
        return 200, b'{"status":"completed","output":[]}'

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    with codex.ResponsesBridge("/fixture-worker.sock") as bridge:
        body = b'{"model":"openai/gpt-5.6-sol","input":"hi"}'
        client = socket.create_connection(("127.0.0.1", bridge.server.server_port))
        client.sendall(
            b"POST /v1/responses HTTP/1.1\r\n"
            + b"Host: 127.0.0.1\r\n"
            + ("Authorization: Bearer " + bridge.token + "\r\n").encode()
            + ("Content-Length: " + str(len(body)) + "\r\n").encode()
            + b"Connection: close\r\n\r\n"
            + body
        )
        assert entered.wait(1)
        client.close()
        assert disconnected.wait(1)
        assert bridge.wait_idle(0.05) is False
        terminal.set()
        assert bridge.wait_idle(1) is True


@pytest.mark.skipif(not os.environ.get("ARENA_TEST_CODEX_BINARY"), reason="set ARENA_TEST_CODEX_BINARY for the real CLI protocol proof")
def test_real_codex_tool_call_and_continuation(monkeypatch, tmp_path):
    request_errors = []
    original_dispatch = codex._dispatch

    def checked_dispatch(socket_path, document, **_kwargs):
        try:
            ops.validate_operation_request("openrouter.responses", document)
        except ops.OperationRequestError as exc:
            request_errors.append((str(exc), list(document), document.get("tools")))
        return original_dispatch(socket_path, document)

    monkeypatch.setattr(codex, "_dispatch", checked_dispatch)
    class ToolTransport(FakeTransport):
        def send(self, **kwargs):
            document = json.loads(kwargs["body"])
            if not self.sent:
                names = [tool["name"] for tool in document["tools"]]
                name = "exec_command" if "exec_command" in names else "shell_command"
                assert name in names, names
                command = "python3 -c 'import lab_arena_codex; from pathlib import Path; Path(\"codex-tool-proof.txt\").write_text(lab_arena_codex.CODEX_VERSION); print(\"ARENA_TOOL_OK\")'"
                arguments = {"cmd": command, "max_output_tokens": 100} if name == "exec_command" else {"command": command}
                self.responses.append((200, response(output=[{"type": "function_call", "id": "fc-1", "call_id": "call-1", "name": name, "arguments": json.dumps(arguments), "status": "completed"}])))
            else:
                assert any(item.get("type") == "function_call_output" and "ARENA_TOOL_OK" in item["output"] for item in document["input"])
                self.responses.append((200, response(id="gen-codex-2")))
            return super().send(**kwargs)

    monkeypatch.setattr(codex, "CODEX_BINARY", os.environ["ARENA_TEST_CODEX_BINARY"])
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    with broker_socket(monkeypatch, ToolTransport()) as (store, transport, path):
        try:
            text = codex.run("Run the shell command provided by the model, then report success.", model="openai/gpt-4o-mini", cwd=tmp_path, timeout_seconds=60)
        except codex.CodexRuntimeError as exc:
            pytest.fail(exc.diagnostics[-8000:] + "\n" + repr(request_errors))
        assert "ARENA_CODEX_OK" in text
        assert (tmp_path / "codex-tool-proof.txt").read_text() == codex.CODEX_VERSION
        assert len(transport.sent) == 2
        assert sum(call["actual"] for call in store.calls.values()) == 24


def _pin_real_codex(monkeypatch):
    binary = os.environ["ARENA_TEST_CODEX_BINARY"]
    assert subprocess.check_output([binary, "--version"], text=True).strip() == "codex-cli 0.154.0"
    monkeypatch.setattr(codex, "CODEX_BINARY", binary)
    monkeypatch.setenv("LAB_ARENA_WORKER_SOCKET", "/unused-worker.sock")
    monkeypatch.setenv("LAB_ARENA_WEB_EGRESS_SOCKET", "/unused-egress.sock")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")


@pytest.mark.skipif(not os.environ.get("ARENA_TEST_CODEX_BINARY"), reason="set ARENA_TEST_CODEX_BINARY to the complete Codex 0.154.0 package")
@pytest.mark.parametrize(
    "statuses,succeeds,expected_posts",
    [
        ((200,), True, 1),
        ((400,), False, 1),
        ((429,), False, 1),
        ((502, 200), False, 1),
        ((502, 502, 200), False, 1),
    ],
    ids=("success", "client-error", "rate-limit", "server-error-recovery", "server-error-cap"),
)
def test_pinned_codex_does_not_automatically_repeat_failed_requests(
    monkeypatch, tmp_path, statuses, succeeds, expected_posts
):
    _pin_real_codex(monkeypatch)
    remaining = list(statuses)
    posts = []

    def dispatch(_socket_path, document, **_kwargs):
        posts.append(document)
        status = remaining.pop(0)
        payload = (
            response()
            if status == 200
            else {"error": {"code": "fixture_error", "message": "bounded fixture failure"}}
        )
        return status, json.dumps(payload, separators=(",", ":")).encode()

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    if succeeds:
        assert "ARENA_CODEX_OK" in codex.run(
            "Reply exactly ARENA_CODEX_OK.",
            model="openai/gpt-4o-mini",
            cwd=tmp_path,
            timeout_seconds=30,
        )
    else:
        with pytest.raises(codex.CodexRuntimeError):
            codex.run(
                "Reply exactly ARENA_CODEX_OK.",
                model="openai/gpt-4o-mini",
                cwd=tmp_path,
                timeout_seconds=30,
            )
    assert len(posts) == expected_posts
    assert len(statuses) - len(remaining) == expected_posts


@pytest.mark.skipif(not os.environ.get("ARENA_TEST_CODEX_BINARY"), reason="set ARENA_TEST_CODEX_BINARY to the complete Codex 0.154.0 package")
def test_pinned_codex_does_not_retry_a_dropped_http_connection(monkeypatch, tmp_path):
    _pin_real_codex(monkeypatch)

    class DropOnceBridge:
        def __init__(self):
            owner = self
            self.posts = 0
            self.token = "drop-once-token"

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_POST(self):
                    size = int(self.headers.get("Content-Length", "0"))
                    self.rfile.read(size)
                    owner.posts += 1
                    if owner.posts == 1:
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        self.connection.close()
                        self.close_connection = True
                        return
                    payload = b"".join(codex.response_events(response()))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)

            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.base_url = "http://127.0.0.1:%d/v1" % self.server.server_address[1]

        def __enter__(self):
            self.thread.start()
            return self

        def wait_idle(self, _timeout_seconds):
            return True

        def __exit__(self, *_args):
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)

    bridge = DropOnceBridge()
    monkeypatch.setattr(codex, "ResponsesBridge", lambda *_args, **_kwargs: bridge)
    with pytest.raises(codex.CodexRuntimeError):
        codex.run(
            "Reply exactly ARENA_CODEX_OK.",
            model="openai/gpt-4o-mini",
            cwd=tmp_path,
            timeout_seconds=30,
        )
    assert bridge.posts == 1


@pytest.mark.skipif(not os.environ.get("ARENA_TEST_CODEX_BINARY"), reason="set ARENA_TEST_CODEX_BINARY to the complete Codex 0.154.0 package")
def test_pinned_codex_does_not_repeat_an_unknown_cost_call(monkeypatch, tmp_path):
    _pin_real_codex(monkeypatch)
    unknown = {"error": {"code": "server_error", "message": "bounded fixture failure"}}
    transport = FakeTransport([(520, unknown), (200, response(id="must-not-dispatch"))])

    with broker_socket(monkeypatch, transport) as (store, transport, _path):
        with pytest.raises(codex.CodexRuntimeError):
            codex.run(
                "Reply exactly ARENA_CODEX_OK.",
                model="openai/gpt-4o-mini",
                cwd=tmp_path,
                timeout_seconds=30,
            )

    assert len(transport.sent) == len(store.calls) == 1
    call = next(iter(store.calls.values()))
    assert call["call_doc"]["action_sequence"] == 0
    assert call["kind"] == "uncertain"
    assert call["uncertain_doc"]["reason"] == "missing_provider_cost"
    assert call["uncertain_doc"]["call_succeeded"] is False


@pytest.mark.skipif(not os.environ.get("ARENA_TEST_CODEX_BINARY"), reason="set ARENA_TEST_CODEX_BINARY to the complete Codex 0.154.0 package")
@pytest.mark.parametrize(
    "limit,expected_reason",
    (("cost", "provider_cost_cap"), ("quota", "per_icp_quota")),
)
def test_pinned_codex_failure_does_not_start_another_billable_request(
    monkeypatch, tmp_path, limit, expected_reason
):
    _pin_real_codex(monkeypatch)

    class OneCallStore(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.action_sequences = []

        def reserve_call(self, **kwargs):
            self.action_sequences.append(kwargs["call_doc"]["action_sequence"])
            if not self.calls:
                if limit == "cost":
                    self.openrouter_capacity = kwargs["amount_microusd"]
                else:
                    self.per_icp_quota = 1
            return super().reserve_call(**kwargs)

    unknown = {
        "error": {"code": "server_error", "message": "bounded fixture failure"}
    }
    store = OneCallStore()
    transport = FakeTransport([(520, unknown), (200, response(id="must-not-dispatch"))])

    with broker_socket(monkeypatch, transport, store=store):
        with pytest.raises(codex.CodexRuntimeError):
            codex.run(
                "Reply exactly ARENA_CODEX_OK.",
                model="openai/gpt-4o-mini",
                cwd=tmp_path,
                timeout_seconds=30,
            )

    assert len(transport.sent) == 1
    assert len(store.calls) == 1
    assert store.action_sequences == [0]
    call = next(iter(store.calls.values()))
    assert call["kind"] == "uncertain"


def test_bridge_worker_broker_chain_uses_the_600_second_responses_timeout(
    monkeypatch,
):
    class SlowCapTransport(FakeTransport):
        def send(self, **kwargs):
            assert 300 < kwargs["timeout_seconds"] <= 600
            return super().send(**kwargs)

    transport = SlowCapTransport([(200, response())])
    with broker_socket(monkeypatch, transport) as (store, transport, path), \
            codex.ResponsesBridge(str(path)) as bridge:
        with httpx.Client(trust_env=False) as client:
            reply = client.post(
                bridge.base_url + "/responses",
                headers={"Authorization": "Bearer " + bridge.token},
                json={
                    "model": "openai/gpt-4o-mini",
                    "input": "hi",
                    "stream": True,
                    "store": False,
                },
            )

    assert reply.status_code == 200
    assert "event: response.completed" in reply.text
    assert len(transport.sent) == len(store.calls) == 1
    assert transport.sent[0]["timeout"] == pytest.approx(600)
    assert next(iter(store.calls.values()))["kind"] == "settlement"


@pytest.mark.skipif(not os.environ.get("ARENA_TEST_CODEX_BINARY"), reason="set ARENA_TEST_CODEX_BINARY to the complete Codex 0.154.0 package")
@pytest.mark.parametrize("model", NATIVE_MODELS)
def test_pinned_codex_selects_each_priced_model_without_runtime_rewrites(monkeypatch, tmp_path, model):
    binary = os.environ["ARENA_TEST_CODEX_BINARY"]
    assert subprocess.check_output([binary, "--version"], text=True).strip() == "codex-cli 0.154.0"
    monkeypatch.setattr(codex, "CODEX_BINARY", binary)

    class FinalTransport(FakeTransport):
        def send(self, **kwargs):
            body = json.loads(kwargs["body"])
            assert body["model"] == model
            self.responses.append((200, response(model=model)))
            return super().send(**kwargs)

    with broker_socket(monkeypatch, FinalTransport(), priced_models=(model,)) as (store, transport, path):
        result = codex.run("Reply exactly ARENA_CODEX_OK.", model=model, cwd=tmp_path, timeout_seconds=60)
    assert "ARENA_CODEX_OK" in result
    assert transport.sent
    assert all(json.loads(sent["body"])["model"] == model for sent in transport.sent)
    assert all(call["kind"] == "settlement" for call in store.calls.values())


def write_local_mcp_fixture(tmp_path):
    """A stdio MCP fixture with two local calls and no provider access."""
    fixture = tmp_path / "arena_fixture_mcp.py"
    fixture.write_text("\n".join([
        "import json, sys",
        "from pathlib import Path",
        "log = Path(sys.argv[1])",
        "tools = [{'name': name, 'description': 'Return a local fixture marker',",
        "          'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False}}",
        "         for name in ('arena_inspect', 'arena_review')]",
        "for line in sys.stdin:",
        "    request = json.loads(line)",
        "    if 'id' not in request: continue",
        "    method = request.get('method')",
        "    if method == 'initialize':",
        "        result = {'protocolVersion': request['params']['protocolVersion'],",
        "                  'capabilities': {'tools': {}}, 'serverInfo': {'name': 'arena-fixture', 'version': '1'}}",
        "    elif method == 'tools/list': result = {'tools': tools}",
        "    elif method == 'tools/call':",
        "        name = request['params']['name']",
        "        assert name in ('arena_inspect', 'arena_review')",
        "        assert request['params'].get('arguments', {}) == {}",
        "        with log.open('a') as output: output.write(name + '\\n')",
        "        result = {'content': [{'type': 'text', 'text': 'ARENA_MCP_' + name.upper() + '_OK'}]}",
        "    else: result = {}",
        "    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)",
    ]) + "\n", encoding="utf-8")
    return fixture


@pytest.mark.skipif(not os.environ.get("ARENA_TEST_CODEX_BINARY"), reason="set ARENA_TEST_CODEX_BINARY to the complete Codex 0.154.0 package")
def test_pinned_luna_code_mode_uses_two_local_mcp_calls_and_compacts(monkeypatch, tmp_path):
    binary = os.environ["ARENA_TEST_CODEX_BINARY"]
    assert subprocess.check_output([binary, "--version"], text=True).strip() == "codex-cli 0.154.0"
    assert Path(binary).resolve().with_name("codex-code-mode-host").is_file()
    monkeypatch.setattr(codex, "CODEX_BINARY", binary)
    fixture = write_local_mcp_fixture(tmp_path)
    mcp_log = tmp_path / "mcp-calls.txt"
    original_session = codex.session

    @contextmanager
    def local_mcp_session(**kwargs):
        with original_session(**kwargs) as environment:
            config = Path(environment["CODEX_HOME"]) / "config.toml"
            # The compaction setting belongs to the top-level TOML table;
            # append only the MCP table after the helper's provider table.
            current = config.read_text(encoding="utf-8")
            setting = (
                "model_auto_compact_token_limit = "
                + str(codex.MODEL_AUTO_COMPACT_TOKEN_LIMIT)
            )
            assert current.count(setting) == 1
            assert current.index(setting) < current.index("[agents]")
            config.write_text(
                current
                + "\n[mcp_servers.fixture]\ncommand = " + json.dumps(sys.executable) + "\n"
                + "args = " + json.dumps([str(fixture), str(mcp_log)]) + "\n"
                + 'required = true\ndefault_tools_approval_mode = "approve"\n',
                encoding="utf-8",
            )
            yield environment

    monkeypatch.setattr(codex, "session", local_mcp_session)

    class NativeTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.requests = []
            self.compaction_requests = 0
            self.tool_calls = 0

        def send(self, **kwargs):
            body = json.loads(kwargs["body"])
            assert body["model"] == "openai/gpt-5.6-luna"
            self.requests.append(body)
            added_tools = any(item.get("type") == "additional_tools" and item.get("tools")
                              for item in body["input"])
            if not self.sent:
                code = "const t = ALL_TOOLS.find(t => t.name.endsWith('arena_inspect')); if (!t) throw new Error('inspect missing'); text(await tools[t.name]({}));"
                self.tool_calls += 1
                output = [{"type": "custom_tool_call", "id": "ct-1", "call_id": "call-1", "name": "exec",
                           "namespace": "functions", "input": code, "status": "completed"}]
            elif not added_tools:
                # Codex's continuation after code-mode output can omit the
                # large additional_tools declaration. Reply with compactable
                # prose; the following request must carry the summary.
                self.compaction_requests += 1
                output = [{"type": "message", "id": "summary-msg", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": "The first local MCP check succeeded. Continue with arena_review.", "annotations": []}]}]
            elif self.tool_calls < 2:
                assert "Another language model started to solve this problem" in json.dumps(body)
                code = "const t = ALL_TOOLS.find(t => t.name.endsWith('arena_review')); if (!t) throw new Error('review missing'); text(await tools[t.name]({}));"
                self.tool_calls += 1
                output = [{"type": "custom_tool_call", "id": "ct-2", "call_id": "call-2", "name": "exec",
                           "namespace": "functions", "input": code, "status": "completed"}]
            else:
                output = [{"type": "message", "id": "final-msg", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": "ARENA_NATIVE_MCP_OK", "annotations": []}]}]
            count = len(self.sent) + 1
            input_tokens = (
                codex.MODEL_AUTO_COMPACT_TOKEN_LIMIT + 1_000 if count == 1 else 100
            )
            usage = {"input_tokens": input_tokens, "output_tokens": 20,
                     "total_tokens": input_tokens + 20, "cost": 0.000012}
            self.responses.append((200, response(model=body["model"], id="native-%d" % count,
                                                 output=output, usage=usage)))
            return super().send(**kwargs)

    with broker_socket(monkeypatch, NativeTransport(), priced_models=("openai/gpt-5.6-luna",)) as (store, transport, path):
        try:
            result = codex.run("Use both local fixture MCP tools, then reply ARENA_NATIVE_MCP_OK.",
                               model="openai/gpt-5.6-luna", reasoning_effort="high", cwd=tmp_path,
                               timeout_seconds=90)
        except codex.CodexRuntimeError as exc:
            pytest.fail(exc.diagnostics[-8000:])
    assert "ARENA_NATIVE_MCP_OK" in result
    assert mcp_log.read_text().splitlines() == ["arena_inspect", "arena_review"]
    assert transport.compaction_requests >= 1, "Usage above the configured limit did not produce a compacted continuation"
    assert any("Another language model started to solve this problem" in json.dumps(body)
               for body in transport.requests[2:])
    assert any(item.get("type") == "custom_tool_call_output" and "input_text" in json.dumps(item.get("output"))
               for body in transport.requests[1:] for item in body["input"])
    assert len(store.calls) == len(transport.sent)
    assert all(call["kind"] == "settlement" for call in store.calls.values())


def test_malformed_reply_fails_closed():
    document = response(output=[{"type": "message", "content": ["bad"]}])
    with pytest.raises(codex.CodexRuntimeError, match="invalid Responses content"):
        list(codex.response_events(document))


def test_noisy_cli_failure_keeps_only_a_bounded_tail(monkeypatch, tmp_path):
    executable = tmp_path / "fake-codex"
    executable.write_text("#!" + sys.executable + "\nimport sys\nsys.stdout.write('x' * 300000)\nsys.exit(7)\n")
    executable.chmod(0o755)
    monkeypatch.setattr(codex, "CODEX_BINARY", str(executable))
    with broker_socket(monkeypatch), pytest.raises(codex.CodexRuntimeError, match="status 7") as error:
        codex.run("test", model="openai/gpt-4o-mini", cwd=tmp_path, timeout_seconds=5)
    assert error.value.diagnostics == "x" * codex.MAX_LOG_BYTES


def test_cli_timeout_stops_the_process(monkeypatch, tmp_path):
    executable = tmp_path / "fake-codex"
    executable.write_text("#!" + sys.executable + "\nimport time\ntime.sleep(60)\n")
    executable.chmod(0o755)
    monkeypatch.setattr(codex, "CODEX_BINARY", str(executable))
    with broker_socket(monkeypatch), pytest.raises(subprocess.TimeoutExpired):
        codex.run("test", model="openai/gpt-4o-mini", cwd=tmp_path, timeout_seconds=0.1)


@pytest.mark.skipif(sys.platform != "linux" or not os.environ.get("ARENA_TEST_CODEX_ROOTFS"), reason="requires Linux root, runsc and the built Arena rootfs")
def test_codex_in_real_gvisor(monkeypatch):
    """Execute the shipped entrypoint/helper/CLI in the actual Arena sandbox."""
    root = Path(__file__).resolve().parents[2]
    tool = {"id": "fc-1", "type": "function_call", "call_id": "call-1", "name": "exec_command", "arguments": json.dumps({"cmd": "python3 -c 'import lab_arena_codex; print(\"ARENA_TOOL_OK\")'", "max_output_tokens": 100}), "status": "completed"}

    class ToolTransport(FakeTransport):
        def send(self, **kwargs):
            if not self.sent:
                self.responses.append((200, response(output=[tool])))
            else:
                body = json.loads(kwargs["body"])
                assert any(item.get("type") == "function_call_output" and "ARENA_TOOL_OK" in item["output"] for item in body["input"])
                self.responses.append((200, response(id="gen-codex-2")))
            return super().send(**kwargs)

    with broker_socket(monkeypatch, ToolTransport()) as (store, transport, socket_path), tempfile.TemporaryDirectory(prefix="arena-codex-gvisor-", dir="/tmp") as directory:
        work = Path(directory)
        work.chmod(0o755)
        for name in ("source", "deps", "input", "output", "runtime"):
            (work / name).mkdir(mode=0o755)
        (work / "input/icp.json").write_text('{"icp":{},"company_limit":5}')
        (work / "source/harness.py").write_text(
            "import lab_arena_codex\n"
            "def run_icp(icp):\n"
            "    result = lab_arena_codex.run('Run the requested tool, then finish.', model='openai/gpt-4o-mini', cwd='/tmp', timeout_seconds=60)\n"
            "    assert 'ARENA_CODEX_OK' in result\n"
            "    return []\n"
        )
        for name in ("agent_entrypoint.py", "lab_arena_checkpoint.py", "lab_arena_codex.py", "web_egress_bridge.py"):
            shutil.copyfile(root / "lab_arena" / name, work / name)
            (work / name).chmod(0o444)
        spec = runtime.SandboxSpec(
            sandbox_id="arena-codex-probe", rootfs_path=Path(os.environ["ARENA_TEST_CODEX_ROOTFS"]),
            input_dir=work / "input", output_dir=work / "output", socket_path=socket_path,
            source_dir=work / "source", dependency_dir=work / "deps",
            agent_entrypoint_path=work / "agent_entrypoint.py", checkpoint_module_path=work / "lab_arena_checkpoint.py",
            codex_module_path=work / "lab_arena_codex.py", web_bridge_path=work / "web_egress_bridge.py",
            entry_command=runtime.AGENT_ENTRY_COMMAND, working_dir=runtime.AGENT_WORKING_DIR,
            evaluation_date="2026-09-15", random_seed=1, wall_clock_seconds=90,
        )
        result = runtime.RunscRuntime(runtime.RuntimeConfig(
            runsc_path=Path(os.environ.get("ARENA_TEST_RUNSC", "/usr/local/bin/runsc")), work_dir=work / "runtime",
        )).run_icp(spec)
        assert result.exit_code == 0 and not result.timed_out, result.stderr.decode(errors="replace")
        assert result.output_bytes == b'{"companies":[]}'
        assert len(transport.sent) == 2
        assert sum(call["actual"] for call in store.calls.values()) == 24
