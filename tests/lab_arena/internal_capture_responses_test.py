"""Internal native search export preserves actual caps, output order, and gaps."""
from copy import deepcopy
import json

import pytest

from lab_arena import internal_capture, operations, training_steps
from lab_arena.service import LocalObjectStore


def request():
    return {"model": "test/model", "input": [
        {"role": "system", "content": "Find a company using public sources."},
        {"role": "user", "content": "Return the supported answer."}],
        "reasoning": {"effort": "high", "context": "all_turns"},
        "max_output_tokens": 16000, "max_tool_calls": 10,
        "tools": [{"type": "openrouter:web_search", "parameters": {
            "engine": "exa", "max_uses": 10, "max_results": 5,
            "max_total_results": 20, "max_characters": 4000}}]}


def response():
    # This is the actual native envelope shape: URL-only action.sources, with
    # later queries present even when no result references were exposed.
    return {"object": "response", "status": "completed", "model": "test/model",
            "usage": {"server_tool_use_details": {"web_search_requests": 2}}, "output": [
        {"type": "reasoning", "id": "r1", "status": "completed", "summary": [],
         "content": [{"type": "reasoning_text", "text": "Choose the first query."}]},
        {"type": "openrouter:web_search", "id": "s1", "status": "completed",
         "action": {"type": "search", "query": "company original source", "sources": [
             {"type": "url", "url": "https://example.com/source"}]}},
        {"type": "reasoning", "id": "r2", "status": "completed", "summary": [],
         "content": [{"type": "reasoning_text", "text": "Check a second query."}]},
        {"type": "openrouter:web_search", "id": "s2", "status": "completed",
         "action": {"type": "search", "query": "company current state"}},
        {"type": "reasoning", "id": "r3", "status": "completed", "summary": [],
         "content": [{"type": "reasoning_text", "text": "Explain the observed result."}]},
        {"type": "message", "id": "answer", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Supported answer.", "annotations": [
             {"type": "url_citation", "url": "https://example.com/source", "title": "Original source",
              "start_index": 0, "end_index": 9}]}]}]}


def export(tmp_path, *, body=None, provider=None):
    objects = LocalObjectStore(tmp_path)
    body, provider = request() if body is None else body, response() if provider is None else provider
    call = internal_capture.InternalCapture(objects).start(
        purpose="icp_generation", context={"set_id": 20261105}, request=body, api="openrouter.responses")
    call.response(body=json.dumps(provider).encode(), status=200)
    call.finish(outcome={"business_outcome": "generated"}, cost_microusd=1)
    return internal_capture.read_call(objects, call.prefix + "request.json")


def test_internal_search_sft_retains_exact_input_and_all_native_output_in_order(tmp_path):
    before_body, before_response = request(), response()
    raw, candidate = export(tmp_path, body=before_body, provider=before_response)
    assert candidate is not None
    assert candidate["input"] == before_body == request()
    assert candidate["target"] == before_response["output"] == response()["output"]
    assert candidate["input"]["max_tool_calls"] == 10
    assert candidate["input"]["tools"][0]["parameters"]["max_total_results"] == 20
    assert [item["id"] for item in candidate["target"]] == ["r1", "s1", "r2", "s2", "r3", "answer"]
    assert len(candidate["available_reasoning"]) == 3
    assert raw["sft_exclusions"] == [] and raw["completeness"]["complete"] is True
    assert candidate["quality"]["status"] == "unreviewed"


def test_url_references_are_not_reported_as_result_content_or_full_evidence(tmp_path):
    raw, candidate = export(tmp_path)
    evidence = raw["research_evidence"]
    assert evidence == candidate["source"]["research_evidence"]
    assert evidence["search_events"] == 2
    assert evidence["events_with_source_references"] == evidence["source_references"] == 1
    assert evidence["events_with_result_text"] == 0
    assert evidence["all_search_events_have_result_text"] is False
    assert evidence["underlying_page_contents_verified"] is False
    assert evidence["all_internal_provider_turns_verified"] is False
    assert [item["result_text_status"] for item in evidence["events"]] == ["not_exposed", "not_exposed"]


def test_actual_provider_overage_is_preserved_without_inventing_budget_authority(tmp_path):
    provider = response()
    search = provider["output"][1]
    provider["output"][1:2] = [dict(deepcopy(search), id="s" + str(i)) for i in range(11)]
    provider["usage"]["server_tool_use_details"]["web_search_requests"] = 12
    raw, candidate = export(tmp_path, provider=provider)
    assert candidate is not None and candidate["input"]["max_tool_calls"] == 10
    assert raw["research_evidence"]["search_events"] == 12
    assert candidate["target"] == provider["output"]
    assert "budget_approved" not in json.dumps(candidate)


@pytest.mark.parametrize("damage", ["incomplete", "error", "errors", "refusal", "message_refusal",
                                    "unfinished_search", "missing_query", "unsafe_source", "malformed_reasoning",
                                    "undeclared_function"])
def test_malformed_refused_and_incomplete_native_results_stay_raw_without_sft(tmp_path, damage):
    provider = response()
    if damage == "incomplete":
        provider["status"] = "incomplete"
    elif damage in {"error", "errors", "refusal"}:
        provider[damage] = {"code": "refused"}
    elif damage == "message_refusal":
        provider["output"][-1]["content"] = [{"type": "refusal", "refusal": "Cannot answer."}]
    elif damage == "unfinished_search":
        provider["output"][1]["status"] = "in_progress"
    elif damage == "missing_query":
        provider["output"][1]["action"].pop("query")
    elif damage == "unsafe_source":
        provider["output"][1]["action"]["sources"][0]["url"] = "file:///private/source"
    elif damage == "malformed_reasoning":
        provider["output"][0]["content"] = [{"type": "input_text", "text": "Not native reasoning."}]
    else:
        provider["output"][-1] = {"type": "function_call", "id": "function", "call_id": "c1",
                                   "name": "not_declared", "arguments": "{}", "status": "completed"}
    raw, candidate = export(tmp_path, provider=provider)
    if damage == "malformed_reasoning":
        assert raw["response"]["output"][0]["content"] == "[REDACTED]"
        assert raw["sft_exclusions"] == ["target_shape_or_completion_unsupported"]
    else:
        assert raw["response"] == provider
        assert raw["sft_exclusions"] == ["target_shape_or_completion_unsupported"]
    assert candidate is None


@pytest.mark.parametrize("damage", ["assistant_history", "unknown_content", "unknown_tool", "bool_limit",
                                    "zero_limit", "missing_limit", "malformed_search_controls", "unknown_search_control"])
def test_internal_initial_input_and_tool_controls_are_validated_without_rewriting(tmp_path, damage):
    body = request()
    if damage == "assistant_history":
        body["input"][0]["role"] = "assistant"
    elif damage == "unknown_content":
        body["input"][0]["content"] = [{"type": "input_image", "image_url": "https://example.com/image"}]
    elif damage == "unknown_tool":
        body["tools"][0]["type"] = "unknown_tool"
    elif damage == "bool_limit":
        body["max_tool_calls"] = True
    elif damage == "zero_limit":
        body["tools"][0]["parameters"]["max_uses"] = 0
    elif damage == "missing_limit":
        body.pop("max_tool_calls")
    elif damage == "malformed_search_controls":
        body["tools"][0]["parameters"] = []
    else:
        body["tools"][0]["parameters"]["unreviewed_control"] = 10
    raw, candidate = export(tmp_path, body=body)
    assert raw["request"] == body and candidate is None


def test_default_shared_miner_search_boundary_stays_at_one_call():
    body, provider = request(), response()
    with pytest.raises(ValueError, match="input_shape_unsupported"):
        training_steps._target(body, provider, "openrouter.responses", allow_reasoning_history=True)
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.responses", {
            "model": body["model"], "input": body["input"], "tools": body["tools"], "max_tool_calls": 10,
        }, allow_full_thinking=True)
    assert operations.OPENROUTER_WEB_SEARCH_MAX_TOOL_CALLS == 1
    allowed = deepcopy(body)
    allowed["max_tool_calls"] = 1
    allowed["tools"][0]["parameters"] = {"engine": "exa", "max_uses": 1, "max_total_results": 5}
    target, _ = training_steps._target(allowed, provider, "openrouter.responses", allow_reasoning_history=True)
    assert target == provider["output"]


def test_callback_does_not_change_chat_validation_or_input_bytes(tmp_path):
    body = {"model": "test/model", "messages": [{"role": "user", "content": "Read source."}]}
    provider = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Answer."}}]}
    def forbidden(*args, **kwargs):
        pytest.fail("Responses callback reached Chat path")
    original = deepcopy(body)
    target, _ = training_steps._target(body, provider, "openrouter.chat", responses_input_validator=forbidden)
    assert target == provider["choices"][0]["message"] and body == original
