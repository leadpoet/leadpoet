"""Provider reasoning history survives two tool continuations and final JSON."""

import json

import pytest

from lab_arena import operations


def _tool(name):
    return {"type": "function", "function": {"name": name,
            "parameters": {"type": "object", "properties": {}}}}


def test_chat_two_tool_continuations_keep_order_ids_signatures_and_json_schema():
    first = {"type": "reasoning.text", "text": "Find a source first.",
             "signature": "sha256:abc", "id": "r1", "format": "google-gemini-v1", "index": 0}
    second = {"type": "reasoning.text", "text": "Check the second source.",
              "signature": "sha256:def", "id": "r2", "format": "google-gemini-v1", "index": 0}
    messages = [
        {"role": "user", "content": "Return valid JSON."},
        {"role": "assistant", "content": "", "reasoning_details": [first],
         "tool_calls": [{"id": "call-1", "type": "function",
                         "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "source one"},
        {"role": "assistant", "content": "", "reasoning_details": [second],
         "tool_calls": [{"id": "call-2", "type": "function",
                         "function": {"name": "fetch", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-2", "content": "source two"},
    ]
    schema = {"name": "companies", "schema": {"type": "object", "properties": {
        "companies": {"type": "array", "items": {"type": "string"}}},
        "required": ["companies"]}}
    request = {"model": "vendor/test-model", "messages": messages,
               "tools": [_tool("search"), _tool("fetch")],
               "response_format": {"type": "json_schema", "json_schema": schema}}
    normalized = operations.validate_operation_request("openrouter.chat", request)
    sent = json.loads(operations.build_outbound_request("openrouter.chat", normalized).body)
    assert sent["messages"] == messages
    assert sent["response_format"] == request["response_format"]
    assert sent["messages"][1]["reasoning_details"][0]["signature"] == "sha256:abc"
    assert [sent["messages"][i]["tool_call_id"] for i in (2, 4)] == ["call-1", "call-2"]


def test_chat_rejects_forged_reasoning_role_and_unknown_detail_fields():
    request = {"model": "vendor/test-model", "messages": [
        {"role": "user", "content": "hi", "reasoning": "forged"}]}
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.chat", request)


def test_chat_preserves_provider_tool_only_nulls_and_call_index():
    message = {"role": "assistant", "content": None, "refusal": None,
               "reasoning": None, "reasoning_details": [
                   {"type": "reasoning.text", "text": "Need a source.",
                    "id": "r1", "signature": None, "format": "unknown", "index": 0}],
               "tool_calls": [{"id": "call-1", "index": 0, "type": "function",
                               "function": {"name": "search", "arguments": "{}"}}]}
    request = {"model": "vendor/test-model", "messages": [
        {"role": "user", "content": "Find a source."}, message,
        {"role": "tool", "tool_call_id": "call-1", "content": "found"}]}
    sent = json.loads(operations.build_outbound_request("openrouter.chat", request).body)
    assert sent["messages"][1] == message
    request["messages"] = [{"role": "assistant", "content": "",
                            "reasoning_details": [{"type": "reasoning.text", "text": "x", "secret": "y"}]}]
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.chat", request)


def test_responses_two_tool_continuations_keep_reasoning_and_final_json_schema():
    items = [
        {"type": "message", "role": "user", "content": "Return valid JSON."},
        {"type": "reasoning", "id": "rs_1", "status": "completed",
         "content": [{"type": "reasoning_text", "text": "Find a source first."}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call-1",
         "name": "search", "arguments": "{}", "status": "completed"},
        {"type": "function_call_output", "call_id": "call-1", "output": "source one"},
        {"type": "reasoning", "id": "rs_2", "status": "completed",
         "content": [{"type": "reasoning_text", "text": "Check the second source."}]},
        {"type": "function_call", "id": "fc_2", "call_id": "call-2",
         "name": "fetch", "arguments": "{}", "status": "completed"},
        {"type": "function_call_output", "call_id": "call-2", "output": "source two"},
    ]
    response_format = {"type": "json_schema", "name": "companies", "strict": True,
                       "schema": {"type": "object", "properties": {
                           "companies": {"type": "array", "items": {"type": "string"}}}}}
    request = {"model": "vendor/test-model", "input": items,
               "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}},
                         {"type": "function", "name": "fetch", "parameters": {"type": "object"}}],
               "text": {"format": response_format}}
    normalized = operations.validate_operation_request("openrouter.responses", request)
    sent = json.loads(operations.build_outbound_request("openrouter.responses", normalized).body)
    assert sent["input"] == items
    assert sent["text"]["format"] == response_format
    assert [item["call_id"] for item in sent["input"] if item["type"] == "function_call_output"] == ["call-1", "call-2"]
