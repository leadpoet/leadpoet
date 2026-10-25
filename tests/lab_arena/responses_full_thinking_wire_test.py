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
