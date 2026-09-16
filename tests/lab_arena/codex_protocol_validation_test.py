"""Closed native Codex Responses shapes at the paid provider boundary."""

import pytest

from lab_arena import contracts, lab_arena_codex as codex, operations


def _schema(levels=4):
    node = {"type": "string"}
    for _ in range(levels):
        node = {"type": "object", "properties": {"child": node}}
    return node


def _local_namespace():
    return {
        "type": "namespace", "name": "functions", "description": "Local client tools",
        "tools": [
            {"type": "function", "name": "wait", "strict": False, "parameters": _schema()},
            {"type": "custom", "name": "exec", "format": {
                "type": "grammar", "syntax": "lark", "definition": "start: SOURCE\nSOURCE: /[^\\r\\n]+/",
            }},
        ],
    }


def _native_input(*, tools=True, call=False):
    items = [
        {"type": "additional_tools", "id": "at_1", "role": "developer", "tools": [_local_namespace()] if tools else []},
        {"type": "message", "id": "msg_1", "role": "user", "content": [{"type": "input_text", "text": "Use the local tool."}]},
    ]
    if call:
        items += [
            {"type": "custom_tool_call", "status": "completed", "call_id": "call-1", "name": "exec", "namespace": "functions", "input": "run"},
            {"type": "custom_tool_call_output", "id": "ctco_1", "call_id": "call-1", "output": [
                {"type": "input_text", "text": "completed"}, {"type": "input_text", "text": "more output"},
            ]},
        ]
    return items


@pytest.mark.parametrize("tools,call", [(True, False), (True, True), (False, True)])
def test_native_standard_namespaced_and_compaction_shapes_preserve_input(tools, call):
    params = {
        "model": "openai/gpt-5.4", "input": _native_input(tools=tools, call=call),
        "reasoning": {"effort": "max", "context": "all_turns"},
        "max_output_tokens": 16_384,
    }
    normalized = operations.validate_operation_request("openrouter.responses", params)
    assert normalized["input"] == params["input"]
    assert normalized["reasoning"] == params["reasoning"]
    assert normalized["max_output_tokens"] == 16_384


def test_assistant_phase_and_typed_function_output_are_preserved():
    items = _native_input()
    items += [
        {"type": "message", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "text": "Checking."}]},
        {"type": "function_call", "call_id": "call-2", "name": "wait", "arguments": "{}", "status": "completed"},
        {"type": "function_call_output", "call_id": "call-2", "output": [{"type": "input_text", "text": "done"}]},
    ]
    params = {"model": "openai/gpt-5.4", "input": items}
    assert operations.validate_operation_request("openrouter.responses", params)["input"] == items


@pytest.mark.parametrize("hosted", ["web_search", "mcp", "shell", "code_interpreter"])
def test_nested_hosted_tools_fail_closed(hosted):
    params = {"model": "openai/gpt-5.4", "input": _native_input()}
    params["input"][0]["tools"][0]["tools"].append({"type": hosted, "name": "hosted"})
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", params)


@pytest.mark.parametrize("output", [[], [{"type": "input_image", "text": "x"}], [{"type": "input_text", "text": 1}], {"type": "input_text", "text": "x"}])
def test_malformed_typed_custom_output_fails_closed(output):
    params = {"model": "openai/gpt-5.4", "input": _native_input(call=True)}
    params["input"][-1]["output"] = output
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", params)


@pytest.mark.parametrize("change", [
    ("role", "tool"), ("role", "assistant"), ("id", 3),
])
def test_additional_tool_item_has_closed_role_and_identifier(change):
    params = {"model": "openai/gpt-5.4", "input": _native_input()}
    params["input"][0][change[0]] = change[1]
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", params)


def test_phase_on_user_message_and_bad_call_status_fail_closed():
    params = {"model": "openai/gpt-5.4", "input": _native_input(call=True)}
    params["input"][1]["phase"] = "final_answer"
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", params)
    params["input"][1].pop("phase")
    params["input"][2]["status"] = "queued"
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", params)


def test_total_local_tool_count_and_schema_depth_are_bounded():
    params = {"model": "openai/gpt-5.4", "input": _native_input()}
    params["input"][0]["tools"][0]["tools"] = [
        {"type": "function", "name": f"tool_{i}"} for i in range(64)
    ]
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", params)
    params = {"model": "openai/gpt-5.4", "input": _native_input()}
    params["input"][0]["tools"][0]["tools"][0]["parameters"] = _schema(30)
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", params)


def test_responses_frame_allowance_is_scoped_and_shared_caps_remain():
    frame = {"operation_id": "openrouter.responses", "parameters": {"model": "openai/gpt-5.4", "input": _native_input()}}
    contracts.check_strict_document(frame, contracts.RESPONSES_PROVIDER_FRAME_LIMITS)
    with pytest.raises(contracts.ArenaContractError):
        contracts.check_strict_document(frame, contracts.PROVIDER_FRAME_LIMITS)
    assert operations.OPERATION_LIMITS.max_depth == 12
    assert operations.RESPONSES_OPERATION_LIMITS.max_depth == 24
    assert operations.OPERATIONS["openrouter.chat"].cost_rule["max_output_tokens"] == 4096
    assert operations.OPERATIONS["openrouter.responses"].cost_rule["max_output_tokens"] == 32_768
    assert operations.operation_table_document()["operation_limit_overrides"]["openrouter.responses"]["max_depth"] == 24


def test_oversized_response_cap_is_rejected_before_provider_call():
    params = {"model": "openai/gpt-5.4", "input": _native_input(), "max_output_tokens": 32_769}
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", params)


def test_encrypted_reasoning_history_uses_structural_string_ceiling():
    params = {"model": "openai/gpt-5.4", "input": [
        {"type": "reasoning", "encrypted_content": "a" * 64_000},
        {"type": "message", "role": "user", "content": "Continue."},
    ]}
    operations.validate_operation_request("openrouter.responses", params)
    params["input"][0]["encrypted_content"] = "a" * 128_001
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", params)


def test_codex_tool_output_chunk_limit_matches_operation_contract():
    assert codex.TOOL_OUTPUT_TEXT_CHARS == operations.OPENROUTER_MAX_CONTENT_CHARS


@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
def test_codex_chunks_only_oversized_tool_output_without_losing_unicode(kind):
    limit = codex.TOOL_OUTPUT_TEXT_CHARS
    text = "a" * (limit - 1) + "🙂β" + "z" * limit
    params = {"model": "openai/gpt-5.4", "input": [
        {"type": kind, "call_id": "call-1", "output": text},
    ]}

    result = codex._chunk_tool_output_text(params)
    parts = result["input"][0]["output"]
    assert [len(part["text"]) for part in parts] == [limit, limit, 1]
    assert "".join(part["text"] for part in parts) == text
    assert codex._chunk_tool_output_text(result) == result
    operations.validate_operation_request("openrouter.responses", result)


def test_codex_chunks_valid_typed_parts_in_order_and_preserves_invalid_parts():
    limit = codex.TOOL_OUTPUT_TEXT_CHARS
    invalid = {"type": "input_text", "text": "u" * (limit + 1), "future": True}
    oversized = "界" * (limit + 1)
    params = {"model": "openai/gpt-5.4", "input": [
        {"type": "custom_tool_call", "call_id": "call-1", "name": "exec",
         "input": "x" * (limit + 1)},
        {"type": "custom_tool_call_output", "call_id": "call-1", "output": [
            {"type": "input_text", "text": "before"},
            {"type": "input_text", "text": oversized},
            invalid,
            {"type": "input_text", "text": "after"},
        ]},
        {"type": "message", "role": "user", "content": "m" * (limit + 1)},
    ]}

    result = codex._chunk_tool_output_text(params)
    output = result["input"][1]["output"]
    assert output[0] == {"type": "input_text", "text": "before"}
    assert [len(output[1]["text"]), len(output[2]["text"])] == [limit, 1]
    assert output[1]["text"] + output[2]["text"] == oversized
    assert output[3] is invalid
    assert output[4] == {"type": "input_text", "text": "after"}
    assert result["input"][0]["input"] == "x" * (limit + 1)
    assert result["input"][2]["content"] == "m" * (limit + 1)
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", result)


@pytest.mark.parametrize("size", [codex.TOOL_OUTPUT_TEXT_CHARS - 1, codex.TOOL_OUTPUT_TEXT_CHARS])
def test_codex_leaves_boundary_tool_output_strings_unchanged(size):
    output = "x" * size
    params = {"input": [{"type": "function_call_output", "call_id": "call-1", "output": output}]}
    assert codex._chunk_tool_output_text(params)["input"][0]["output"] is output


def test_codex_worker_frame_cap_still_rejects_chunked_tool_output():
    params = {"input": [{
        "type": "custom_tool_call_output", "call_id": "call-1",
        "output": "x" * 1_048_576,
    }]}
    codex._chunk_tool_output_text(params)
    with pytest.raises(codex.CodexRuntimeError, match="request too large"):
        codex._dispatch("/not-used", params)
