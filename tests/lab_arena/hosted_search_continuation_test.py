"""Shape-faithful hosted Responses replay; no provider calls or credentials."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import httpx
import pytest

from lab_arena import broker, lab_arena_codex as codex, operations, reasoning_capture
from tests.lab_arena.codex_runtime_test import _pin_real_codex, response
from tests.lab_arena.reasoning_capture_test import _reviewed_registry
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker


def hosted_output():
    # Shapes observed in the real 2026-10-25 wire capture. Content is synthetic.
    return [
        {"id": "reason-search", "type": "reasoning", "status": "completed",
         "summary": [], "content": [{"type": "reasoning_text", "text": "Need current evidence."}]},
        {"id": "msg-search", "type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Checking sources.", "annotations": [], "logprobs": []}]},
        {"id": "st-source", "type": "openrouter:web_search", "status": "completed",
         "action": {"type": "search", "query": "current fixture revenue",
                    "sources": [{"type": "url", "url": "https://example.com/source"}]}},
        {"id": "st-empty", "type": "openrouter:web_search", "status": "completed",
         "action": {"type": "search", "query": "fixture confirmation"}},
        {"id": "msg-cited", "type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Found a source.", "logprobs": [],
                      "annotations": [{"type": "url_citation", "url": "https://example.com/source",
                                       "title": "Source", "content": "Source excerpt.",
                                       "start_index": 8, "end_index": 14}]}]},
    ]


def hosted_request():
    return {"model": "openai/gpt-4o-mini", "provider": {"only": ["example/verified"]}, "input": [
        {"type": "message", "role": "user", "content": "Find evidence."},
        *hosted_output(),
        {"type": "function_call", "id": "fc-local", "call_id": "local-1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "local-1", "output": "local evidence"},
    ], "tools": [{"type": "openrouter:web_search", "parameters": {
        "engine": "native", "max_uses": operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS,
        "max_total_results": operations.OPENROUTER_WEB_SEARCH_MAX_TOTAL_RESULTS}}],
        "max_tool_calls": operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS,
        "text": {"format": {"type": "json_schema", "name": "result", "strict": True,
                            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                       "required": ["ok"], "additionalProperties": False}}}}


def test_full_hosted_continuation_preserves_native_output_and_local_json():
    request = hosted_request()
    normalized = broker.normalized_request("openrouter.responses", request,
        reasoning_capture_policy=reasoning_capture.POLICY,
        reasoning_route_registry=_reviewed_registry(api="openrouter.responses"))
    assert normalized["input"] == request["input"]
    assert normalized["tools"] == request["tools"]
    assert normalized["text"] == request["text"]
    assert normalized["max_output_tokens"] == 8192
    assert normalized["provider"]["only"] == ["example/verified"]


def test_policy_off_still_rejects_new_provider_item():
    request = {"model": "openai/gpt-4o-mini", "input": [hosted_output()[2]]}
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", request)


@pytest.mark.parametrize("change", [
    {"type": "openrouter:shell"}, {"id": ""}, {"status": "unknown"}, {"unexpected": "data"},
    {"action": {"type": "open_page", "url": "https://example.com"}},
    {"action": {"type": "search", "query": "x", "sources": [{"type": "url", "url": "file:///etc/passwd"}]}},
    {"action": {"type": "search", "query": "x", "extra": "data"}},
    {"action": {"type": "search"}},
    {"action": {"type": "search", "query": "x", "sources": [{"type": "url", "url": "https://example.com"}] * 129}},
])
def test_full_hosted_continuation_rejects_unknown_or_unbounded_shapes(change):
    item = dict(hosted_output()[2], **change)
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses",
            {"model": "openai/gpt-4o-mini", "input": [item]}, allow_full_thinking=True)


@pytest.mark.parametrize("policy", [None, "", "other", reasoning_capture.POLICY])
@pytest.mark.parametrize("stream", [False, True])
def test_bridge_search_replay_matches_frozen_policy(monkeypatch, policy, stream):
    if policy is None:
        monkeypatch.delenv("LAB_ARENA_REASONING_CAPTURE_POLICY", raising=False)
    else:
        monkeypatch.setenv("LAB_ARENA_REASONING_CAPTURE_POLICY", policy)
    original = response(output=hosted_output())
    snapshot = copy.deepcopy(original)
    raw = json.dumps(original, separators=(",", ":")).encode()
    monkeypatch.setattr(codex, "_dispatch", lambda *_args, **_kwargs: (200, raw))
    with codex.ResponsesBridge("/unused-policy-test.sock", web_search="live") as bridge:
        with httpx.Client(trust_env=False) as client:
            result = client.post(bridge.base_url + "/responses",
                headers={"Authorization": "Bearer " + bridge.token},
                json={"model": "openai/gpt-4o-mini", "input": "Find evidence.", "stream": stream})
    assert result.status_code == 200
    if stream:
        # The default encoder is the historical replay: no provider type rewrite.
        enabled = policy == reasoning_capture.POLICY
        expected = b"".join(codex.response_events(original, native_web_search=enabled))
        assert result.content == expected
        completed = [json.loads(line[6:]) for line in result.content.splitlines()
                     if line.startswith(b"data: ")][-1]["response"]
        expected_type = "web_search_call" if enabled else "openrouter:web_search"
        assert [item["type"] for item in completed["output"]
                if item["id"].startswith("st-")] == [expected_type, expected_type]
    else:
        assert result.content == raw
    assert original == snapshot


@pytest.mark.parametrize("full_policy", [False, True])
def test_standalone_search_projection_is_independent_of_policy(monkeypatch, full_policy):
    if full_policy:
        monkeypatch.setenv("LAB_ARENA_REASONING_CAPTURE_POLICY", reasoning_capture.POLICY)
    else:
        monkeypatch.delenv("LAB_ARENA_REASONING_CAPTURE_POLICY", raising=False)
    original = response(output=hosted_output())
    monkeypatch.setattr(codex, "_dispatch", lambda *_args, **_kwargs:
                        (200, json.dumps(original).encode()))
    with codex.ResponsesBridge("/unused-policy-test.sock", web_search="live") as bridge:
        with httpx.Client(trust_env=False) as client:
            result = client.post(bridge.base_url + "/alpha/search",
                headers={"Authorization": "Bearer " + bridge.token},
                json={"model": "openai/gpt-4o-mini", "commands": {"search_query": [{"q": "fixture"}]},
                      "settings": {"external_web_access": True}})
    assert result.status_code == 200
    assert result.content == codex._standalone_search_output(original)
    assert "https://example.com/source" in result.json()["output"]


@pytest.mark.parametrize("full_policy", [False, True])
def test_policy_hint_works_in_isolated_staged_module(tmp_path, full_policy):
    # Production mounts this file without the lab_arena package. Exercise that
    # same import boundary and the existing runner/shim policy vocabulary.
    from lab_arena import shim
    staged = tmp_path / "lab_arena_codex.py"
    staged.write_bytes(Path(codex.__file__).read_bytes())
    payload = response(output=hosted_output())
    script = """
import importlib.util,json,pathlib,sys,urllib.request
path=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('staged_codex',path)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
document=json.loads(sys.stdin.read())
c._dispatch=lambda *_args,**_kwargs:(200,json.dumps(document).encode())
with c.ResponsesBridge('/unused-policy-test.sock',web_search='live') as bridge:
    request=urllib.request.Request(bridge.base_url+'/responses',
        data=json.dumps({'model':'openai/gpt-4o-mini','input':'fixture','stream':True}).encode(),
        headers={'Authorization':'Bearer '+bridge.token,'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request) as reply:
        result=reply.read()
assert 'lab_arena' not in sys.modules
sys.stdout.buffer.write(result)
"""
    environment = {"PATH": os.environ.get("PATH", "")}
    if full_policy:
        environment[shim.REASONING_POLICY_ENV] = shim.FULL_THINKING_POLICY
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(staged)],
        input=json.dumps(payload).encode(), capture_output=True, env=environment, timeout=10, check=True)
    assert result.stdout == b"".join(codex.response_events(payload, native_web_search=full_policy))


@pytest.mark.skipif(not os.environ.get("ARENA_TEST_CODEX_BINARY"), reason="requires pinned Codex 0.154.0")
@pytest.mark.parametrize("full_policy", [False, True])
def test_pinned_codex_replays_hosted_search_with_native_history(monkeypatch, tmp_path, full_policy):
    _pin_real_codex(monkeypatch)
    if full_policy:
        monkeypatch.setenv("LAB_ARENA_REASONING_CAPTURE_POLICY", reasoning_capture.POLICY)
    else:
        monkeypatch.delenv("LAB_ARENA_REASONING_CAPTURE_POLICY", raising=False)
    posts = []
    items = hosted_output()
    items.append({"id": "ct-local", "type": "custom_tool_call", "status": "completed",
                  "call_id": "call-local", "name": "exec", "namespace": "functions",
                  "input": 'text("LOCAL_OK")'})

    def dispatch(_socket_path, parameters, **_kwargs):
        posts.append(copy.deepcopy(parameters))
        if len(posts) == 1:
            return 200, json.dumps(response(output=items)).encode()
        # This is the exact history sent onward to the provider, after Codex's
        # typed parser and local tool execution. No fixtures substitute it.
        searches = [item for item in parameters["input"] if item.get("type") == "web_search_call"]
        expected_searches = [
            {"type": "search", "query": "current fixture revenue"},
            {"type": "search", "query": "fixture confirmation"},
        ]
        # Historical Codex drops the unrecognized provider item. Only opted-in
        # execution receives the native type that its typed history preserves.
        assert [item["action"] for item in searches] == (expected_searches if full_policy else [])
        # Codex's existing typed serializer omits sources, annotations and IDs.
        # Do not change agent behavior by injecting those fields back in.
        messages = [item for item in parameters["input"] if item.get("role") == "assistant"]
        assert any(item["content"] == [{"type": "output_text", "text": "Found a source."}] for item in messages)
        assert all("id" not in item for item in searches)
        assert not any(item.get("type") == "openrouter:web_search" for item in parameters["input"])
        assert any(item.get("type") == "custom_tool_call_output" for item in parameters["input"])
        operations.validate_operation_request("openrouter.responses", parameters, allow_full_thinking=True)
        return 200, json.dumps(response()).encode()

    monkeypatch.setattr(codex, "_dispatch", dispatch)
    result = codex.run("Search then use the local tool.", model="openai/gpt-4o-mini", cwd=tmp_path,
                       timeout_seconds=30, web_search="live")
    assert "ARENA_CODEX_OK" in result
    assert len(posts) == 2


def test_native_stream_adaptation_preserves_raw_provider_receipt():
    original = response(output=hosted_output())
    snapshot = copy.deepcopy(original)
    def completed(data):
        return [json.loads(line[6:]) for line in data.splitlines() if line.startswith(b"data: ")][-1]["response"]
    raw = completed(b"".join(codex.response_events(original)))
    native = completed(b"".join(codex.response_events(original, native_web_search=True)))
    assert raw == snapshot == original
    expected = copy.deepcopy(original)
    for item in expected["output"]:
        if item["type"] == "openrouter:web_search":
            item["type"] = "web_search_call"
    assert native == expected
    assert original == snapshot


def test_full_hosted_broker_preserves_receipts_privacy_and_budget_settlement():
    registry = _reviewed_registry(api="openrouter.responses")
    context = replace(CONTEXT, reasoning_capture_policy=reasoning_capture.POLICY,
                      reasoning_route_registry=registry)
    first_reply = response(output=hosted_output(), usage={"input_tokens": 10, "output_tokens": 10,
                                                       "total_tokens": 20, "cost": "0.03125"})
    transport = FakeTransport([(200, first_reply), (200, response(id="gen-final"))])
    service, store, transport = make_broker(transport=transport)
    continuation = hosted_request()
    initial = dict(continuation, input=continuation["input"][:1])
    results = [service.execute(context, operation_id="openrouter.responses", parameters=request,
                               action_sequence=index, timeout_ms=300_000)
               for index, request in enumerate((initial, continuation))]
    assert all(result.status == 200 for result in results)
    assert json.loads(results[0].body) == first_reply
    assert store.log == ["reserve", "dispatch", "settle"] * 2
    assert results[0].call["actual_microusd"] == 31_250
    for request, result, sent in zip((initial, continuation), results, transport.sent):
        outgoing = json.loads(sent["body"])
        assert outgoing["input"] == request["input"]
        assert outgoing["tools"] == request["tools"]
        assert outgoing["max_tool_calls"] == request["max_tool_calls"]
        assert outgoing["max_output_tokens"] == 8192
        assert outgoing["text"] == request["text"]
        assert outgoing["provider"]["only"] == ["example/verified"]
        assert outgoing["provider"]["zdr"] is True
        assert outgoing["provider"]["data_collection"] == "deny"
        assert outgoing["provider"]["allow_fallbacks"] is False
        assert outgoing["provider"]["require_parameters"] is True
        assert result.call["reservation_basis"] == "remaining_budget_native_web_search"
    assert results[1].call["reserved_microusd"] == results[0].call["reserved_microusd"] - 31_250
