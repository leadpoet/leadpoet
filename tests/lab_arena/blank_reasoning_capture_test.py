"""Blank provider plaintext is exact evidence, but is not full thinking."""
from copy import deepcopy
import base64
import json

import pytest

from lab_arena import contracts, internal_capture, training_capture, training_export, training_steps, training_thinking
from lab_arena.service import LocalObjectStore
from tests.lab_arena.training_cap_eof_test import cap_example
from tests.lab_arena.training_eligibility_test import refresh_response_links


def reasoning(text):
    # Reduced from the retained November 8 provider item. No private values.
    return {"type": "reasoning", "id": "r-blank", "status": "completed", "summary": [],
            "content": [{"type": "reasoning_text", "text": text}]}


@pytest.mark.parametrize("text", ["", " ", "\n", "\t \r\n", "\u2003"])
def test_blank_native_reasoning_remains_exact_in_raw_history_and_export(text):
    item = reasoning(text)
    response = {"object": "response", "status": "completed", "output": [item]}
    history = {"operation_id": "openrouter.responses", "outbound": {"body": {"input": [item]}}}
    for value in (response, history, {"provider_calls": [{"response": response, "request": history}]}):
        before = deepcopy(value)
        assert training_capture.sanitize_training_value(value) == (value, [])
        assert training_export.without_hidden_reasoning(value) == (value, False)
        assert value == before
    assert training_thinking._segments(response, "openrouter.responses") == ([], False, False)


@pytest.mark.parametrize("payload", [[], [{"type": "reasoning_text", "text": ""}],
                                    [{"type": "reasoning_text", "text": " "},
                                     {"type": "reasoning_text", "text": "Visible thinking."}]])
def test_empty_and_mixed_plaintext_parts_are_kept(payload):
    item = dict(reasoning(""), content=payload)
    value = {"output": [item]}
    assert training_capture.sanitize_training_value(value) == (value, [])
    assert training_export.without_hidden_reasoning(value) == (value, False)
    segments, _, _ = training_thinking._segments(value, "openrouter.responses")
    assert [segment["text"] for segment in segments] == (["Visible thinking."] if len(payload) == 2 else [])


@pytest.mark.parametrize("kind,field", [("reasoning.text", "text"), ("thinking", "thinking"),
                                       ("reasoning.summary", "summary"), ("reasoning_text", "text")])
def test_blank_typed_chat_details_are_plaintext_without_thinking(kind, field):
    value = {"choices": [{"message": {"reasoning_details": [{"type": kind, field: " \n"}]}}]}
    assert training_capture.sanitize_training_value(value) == (value, [])
    assert training_export.without_hidden_reasoning(value) == (value, False)
    assert training_thinking._segments(value, "openrouter.chat")[0] == []


def test_blank_response_projection_passes_but_positive_reasoning_usage_is_missing_thinking():
    example = cap_example("openrouter.responses")
    call = example["provider_calls"][0]
    call["response"]["output"][0] = reasoning(" ")
    call["response"]["output"][1]["arguments"] = "{}"
    call["response"]["usage"]["output_tokens"] = 50
    body = call["request"]["outbound"]["body"]
    expected = training_steps._target(body, call["response"], call["operation_id"])
    clean, redactions = training_capture.sanitize_training_value(call["response"])
    assert redactions == []
    assert training_steps._target(body, clean, call["operation_id"]) == expected
    assert contracts.document_hash(clean) == contracts.document_hash(call["response"])
    call["response"] = clean
    coverage = training_thinking.attempt_coverage(call, "miner", example["configuration"]["reasoning_route_registry"])
    assert coverage["returned"] == "missing_thinking"
    assert coverage["segment_count"] == coverage["characters"] == 0
    refresh_response_links(example, 0)
    trace = training_thinking.readable_trace(example)
    events = trace["calls"][0]["events"]
    assert any(event.get("kind") == "reasoning_item" and event.get("value") == reasoning(" ")
               for event in events)
    full, status = training_thinking.full_text_segments(call, "miner", example["configuration"]["reasoning_route_registry"])
    assert full == [] and status["returned"] == "missing_thinking"


@pytest.mark.parametrize("extra", [{"encrypted_content": "sealed-private-state"},
                                   {"hidden_reasoning": "hidden-private-state"},
                                   {"signature": "private-signature"},
                                   {"unknown_payload": {"data": "opaque-private-state"}}])
def test_blank_plaintext_does_not_disable_opaque_field_redaction(extra):
    item = dict(reasoning(" "), **extra)
    clean, reasons = training_capture.sanitize_training_value(item)
    assert reasons == ["hidden_reasoning"]
    assert clean["content"] == item["content"]
    assert all(clean[key] == "[REDACTED]" for key in extra)
    projected, removed = training_export.without_hidden_reasoning({"output": [item]})
    assert removed and all(key not in projected["output"][0] for key in extra)
    assert projected["output"][0]["content"] == item["content"]


@pytest.mark.parametrize("content", [{"sealed": "opaque-state"},
    [{"type": "reasoning.encrypted", "text": "opaque-state"}],
    [{"type": "reasoning_text", "text": " "}, {"type": "sealed", "data": "opaque-state"}],
    [{"type": "reasoning_text", "text": {"sealed": "opaque-state"}}]])
def test_malformed_or_opaque_content_stays_private(content):
    item = dict(reasoning(" "), content=content)
    clean, reasons = training_capture.sanitize_training_value(item)
    assert clean["content"] == "[REDACTED]" and reasons == ["hidden_reasoning"]
    projected, removed = training_export.without_hidden_reasoning({"output": [item]})
    assert removed and "opaque-state" not in str(projected)


def test_blank_mixed_with_credentials_still_scans_every_leaf():
    item = reasoning(" ")
    item["content"].append({"type": "reasoning_text", "text": "synthetic-private-token"})
    item["api_key"] = "another-private-token"
    clean, reasons = training_capture.sanitize_training_value(item, secrets=("synthetic-private-token",))
    assert clean["content"][0]["text"] == " "
    assert clean["content"][1]["text"] == clean["api_key"] == "[REDACTED]"
    assert reasons == ["credential"]


def test_opaque_typed_item_with_blank_text_is_still_opaque():
    value = {"choices": [{"message": {"reasoning_details": [
        {"type": "reasoning.encrypted", "text": " ", "data": "sealed-private-state"}]}}]}
    clean, reasons = training_capture.sanitize_training_value(value)
    assert reasons == ["hidden_reasoning"]
    assert clean["choices"][0]["message"]["reasoning_details"] == ["[REDACTED]"]
    assert training_export.without_hidden_reasoning(value) == (
        {"choices": [{"message": {"reasoning_details": []}}]}, True)


def test_capture_and_terminal_export_keep_blank_native_history_exact(tmp_path):
    item = reasoning(" ")
    request = {"model": "test/model", "input": [item, {"role": "user", "content": "Continue."}],
               "max_output_tokens": 4096}
    document = {"operation_id": "openrouter.responses", "outbound": {"body": request}}
    objects = LocalObjectStore(tmp_path)
    pointer = training_capture.RequestCapture(objects)(document)
    assert pointer["status"] == "stored" and not pointer["redacted"] and pointer["redactions"] == []
    assert training_capture.read_request(objects, pointer)["outbound"]["body"] == request
    response = {"object": "response", "status": "completed", "output": [item,
        {"type": "message", "role": "assistant", "status": "completed", "content": [
            {"type": "output_text", "text": "Supported answer."}]}]}
    raw = json.dumps(response).encode()
    assert training_export._response_value({"body_b64": base64.b64encode(raw).decode()}, "openrouter") == (response, [])
    internal_request = dict(request, input=[{"role": "user", "content": "Continue."}])
    call = internal_capture.InternalCapture(objects).start(
        purpose="admission_review", context={}, request=internal_request, api="openrouter.responses")
    call.response(body=raw, status=200, elapsed_seconds=0.25)
    call.finish(outcome={"host_decision": {"passed": True}}, cost_microusd=12)
    saved, candidate = internal_capture.read_call(objects, call.prefix + "request.json")
    assert saved["request"] == candidate["input"] == internal_request
    assert saved["response"] == response and saved["sft_exclusions"] == []
    assert candidate["available_reasoning"] == candidate["generated_thinking"] == []
