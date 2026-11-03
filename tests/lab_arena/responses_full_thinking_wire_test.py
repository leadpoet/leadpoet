"""Real provider shapes, reduced to synthetic public protocol fixtures."""
from copy import deepcopy
import pytest
from lab_arena import operations, reasoning_capture, training_capture, training_export


def test_response_settings_are_not_opaque_reasoning():
    value = {"object": "response", "output": [], "reasoning": {"effort": "high"}}
    clean, redactions = training_capture.sanitize_training_value(value)
    assert clean == value and redactions == []
    assert training_export.without_hidden_reasoning(value) == (value, False)


def test_arbitrary_reasoning_object_is_still_private():
    value = {"object": "response", "output": [], "reasoning": {"secret_thought": "opaque"}}
    clean, redactions = training_capture.sanitize_training_value(value)
    assert clean["reasoning"] == "[REDACTED]"
    assert redactions == ["hidden_reasoning"]


def test_exported_native_history_keeps_strict_training_identity():
    from lab_arena import contracts, training_steps, training_thinking
    from tests.lab_arena.training_cap_eof_test import cap_example
    from tests.lab_arena.training_eligibility_test import refresh_response_links
    from tests.lab_arena.training_steps_test import seal

    example = cap_example("openrouter.responses")
    call = example["provider_calls"][0]
    call["response"]["output"][1]["arguments"] = "{}"
    call["response"]["usage"]["output_tokens"] = 50
    body = call["request"]["outbound"]["body"]
    body["input"].append({"type": "reasoning", "id": "r-history", "summary": [],
                          "encrypted_content": None,
                          "content": [{"type": "reasoning_text", "text": "Read the company source."}]})
    request_hash = contracts.document_hash(body)
    call["request_hash"] = call["request"]["request_hash"] = request_hash
    call["request"]["admitted_parameters"] = deepcopy(body)
    call["request"]["hashed_parameters"] = deepcopy(body)
    call["request"]["outbound"]["body_sha256"] = request_hash
    call["lifecycle"][0]["accounting"]["request_hash"] = request_hash
    refresh_response_links(example, 0)
    projected, removed = training_export.without_hidden_reasoning(example)
    assert removed is False
    seal(projected)
    candidate = training_steps._candidate(projected, projected["provider_calls"][0])
    assert candidate["input"]["input"] == body["input"]
    splits, _ = training_thinking.build([projected], training_steps.Rules(production_rounds_only=False))
    assert any(row["source_refs"][0]["call_identity"] == call["call_identity"]
               for row in splits["train"] + splits["eval"])
    projected["provider_calls"][0]["request"]["hashed_parameters"]["input"][-1]["summary"] = None
    with pytest.raises(ValueError, match="request_identity_mismatch"):
        training_steps._candidate(projected, projected["provider_calls"][0])


def test_returned_empty_logprobs_survive_full_thinking_tool_continuation_only():
    message = {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Checking next.", "annotations": [], "logprobs": []}]}
    request = {"model": "deepseek/test", "input": [message], "max_output_tokens": 4096}
    assert operations.validate_operation_request("openrouter.responses", request, allow_full_thinking=True) == request
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", request)


def test_responses_route_requires_exact_selected_endpoint_but_not_chat_only_top_provider():
    annotation = {"model_id": "deepseek/test", "model_version": "deepseek/test-20261025",
                  "provider_route": "fireworks", "provider_name": "Fireworks", "api": "openrouter.responses"}
    wire = {"model": "deepseek/test", "openrouter_metadata": {"endpoints": {"available": [
        {"selected": True, "model": "deepseek/test-20261025", "provider": "Fireworks"}]}}}
    assert reasoning_capture.route_observation(wire, annotation)["status"] == "matched"
    assert reasoning_capture.route_observation(wire, dict(annotation, api="openrouter.chat"))["status"] == "unverified"
    assert reasoning_capture.route_observation(dict(wire, provider="Other"), annotation)["status"] == "mismatch"
    bad = deepcopy(wire)
    bad["openrouter_metadata"]["endpoints"]["available"][0]["provider"] = "Other"
    assert reasoning_capture.route_observation(bad, annotation)["status"] == "mismatch"
    bad["openrouter_metadata"]["endpoints"]["available"].append(deepcopy(bad["openrouter_metadata"]["endpoints"]["available"][0]))
    assert reasoning_capture.route_observation(bad, annotation)["status"] == "unverified"
