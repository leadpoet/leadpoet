"""Internal Chat citations retain exact excerpts without granting miner authority."""
from copy import deepcopy
import json

import pytest

from lab_arena import internal_capture, training_steps
from lab_arena.service import LocalObjectStore


def request():
    return {"model": "test/model", "messages": [{"role": "user", "content": "Find supported facts."}],
            "max_tool_calls": 10, "tools": [{"type": "openrouter:web_search", "parameters": {
                "engine": "exa", "mode": "fast", "max_uses": 10, "max_results": 5,
                "max_total_results": 50, "max_characters": 2000}}]}


def response(count=2):
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": "Supported answer.", "refusal": None,
        "reasoning": "Compare the supplied public citations.",
        "annotations": [{"type": "url_citation", "url_citation": {
            "url": "https://example.com/source-" + str(i), "title": "Source " + str(i),
            "content": "Actual provider excerpt " + str(i), "start_index": 0, "end_index": 0,
        }} for i in range(count)]}}],
        "usage": {"server_tool_use_details": {"web_search_requests": 12}}}


def export(tmp_path, body=None, provider=None, secrets=()):
    body = request() if body is None else body
    provider = response() if provider is None else provider
    objects = LocalObjectStore(tmp_path)
    call = internal_capture.InternalCapture(objects).start(
        purpose="icp_generation", context={"set_id": 20261106}, request=body, secrets=secrets)
    call.response(body=json.dumps(provider).encode(), status=200)
    call.finish(outcome={"business_outcome": "generated"}, cost_microusd=1)
    return internal_capture.read_call(objects, call.prefix + "request.json")


def test_internal_chat_preserves_fifty_nested_annotations_and_zero_offsets(tmp_path):
    body, provider = request(), response(50)
    raw, candidate = export(tmp_path, body, provider)
    assert candidate is not None
    assert raw["request"] == candidate["input"] == body
    assert raw["response"] == provider
    expected = deepcopy(provider["choices"][0]["message"])
    expected.pop("reasoning")
    expected.pop("refusal")
    assert candidate["target"] == expected
    assert len(candidate["target"]["annotations"]) == 50
    assert candidate["quality"] == {"status": "unreviewed", "correct_action": "not_established"}
    assert candidate["training_permission"]["status"] == "confirmed"


def test_chat_summary_keeps_citations_and_aggregate_separate_from_search_events(tmp_path):
    provider = response()
    raw, candidate = export(tmp_path, provider=provider)
    evidence = raw["research_evidence"]
    citations = evidence["chat_url_citations"]
    assert citations["references"] == citations["annotations_with_content"] == 2
    assert citations["content_characters"] == sum(len(a["url_citation"]["content"])
                                                   for a in provider["choices"][0]["message"]["annotations"])
    assert citations["items"][0] == {
        "annotation_path": "/choices/0/message/annotations/0",
        "url_path": "/choices/0/message/annotations/0/url_citation/url",
        "content_path": "/choices/0/message/annotations/0/url_citation/content",
        "content_characters": len("Actual provider excerpt 0"),
    }
    assert evidence["provider_reported_web_search_requests"]["count"] == 12
    assert evidence["search_events"] == evidence["source_references"] == 0
    assert evidence["all_search_events_have_result_text"] is False
    assert evidence["underlying_page_contents_verified"] is False
    assert evidence["all_internal_provider_turns_verified"] is False
    assert evidence == candidate["source"]["research_evidence"]
    assert "Actual provider excerpt" not in json.dumps(evidence)
    assert "https://example.com" not in json.dumps(evidence)


def test_url_only_and_blank_content_are_not_reported_as_snippets(tmp_path):
    provider = response()
    annotations = provider["choices"][0]["message"]["annotations"]
    annotations[0]["url_citation"].pop("content")
    annotations[1]["url_citation"]["content"] = "  "
    raw, candidate = export(tmp_path, provider=provider)
    assert candidate is not None
    citations = raw["research_evidence"]["chat_url_citations"]
    assert citations["references"] == 2 and citations["annotations_with_content"] == 0
    assert all(item["content_path"] is None for item in citations["items"])


@pytest.mark.parametrize("value", [True, False, -1, "12", 1.5, {}, None])
def test_invalid_provider_search_counts_are_not_inferred(value):
    provider = response()
    provider["usage"]["server_tool_use_details"].update(
        web_search_requests=value, tool_calls_executed=12)
    assert internal_capture._search_evidence(provider)["provider_reported_web_search_requests"]["count"] is None


def test_reported_zero_and_missing_aggregate_are_distinct():
    provider = response()
    provider["usage"]["server_tool_use_details"]["web_search_requests"] = 0
    assert internal_capture._search_evidence(provider)["provider_reported_web_search_requests"]["count"] == 0
    provider.pop("usage")
    assert internal_capture._search_evidence(provider)["provider_reported_web_search_requests"]["count"] is None
    assert internal_capture._search_evidence({"choices": [{"message": {
        "content": "See https://example.com", "reasoning": "Search excerpt in thinking"}}]})["chat_url_citations"]["references"] == 0


@pytest.mark.parametrize("damage", ["file_url", "http_url", "ip_url", "userinfo_url", "fragment_url",
                                    "unknown_outer", "unknown_nested", "wrong_type", "not_mapping",
                                    "bad_title", "bad_content", "negative_offset", "bool_offset", "too_many",
                                    "refusal", "incomplete", "provider_error", "tool_call"])
def test_invalid_chat_citations_stay_raw_without_sft(tmp_path, damage):
    provider = response()
    message = provider["choices"][0]["message"]
    annotation = message["annotations"][0]
    citation = annotation["url_citation"]
    urls = {"file_url": "file:///private/source", "http_url": "http://example.com/source",
            "ip_url": "https://127.0.0.1/source", "userinfo_url": "https://user:password@example.com/source",
            "fragment_url": "https://example.com/source#fragment"}
    if damage in urls:
        citation["url"] = urls[damage]
    elif damage == "unknown_outer": annotation["opaque"] = "unknown"
    elif damage == "unknown_nested": citation["unreviewed"] = "unknown"
    elif damage == "wrong_type": annotation["type"] = "unknown"
    elif damage == "not_mapping": annotation["url_citation"] = "not a citation"
    elif damage == "bad_title": citation["title"] = 1
    elif damage == "bad_content": citation["content"] = {"text": "unsupported"}
    elif damage == "negative_offset": citation["start_index"] = -1
    elif damage == "bool_offset": citation["end_index"] = True
    elif damage == "too_many": message["annotations"] *= 65
    elif damage == "refusal": message["refusal"] = "Cannot answer."
    elif damage == "incomplete": provider["choices"][0]["finish_reason"] = "length"
    elif damage == "provider_error": provider["error"] = {"code": "failed"}
    else: message["tool_calls"] = [{"id": "invented", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
    raw, candidate = export(tmp_path, provider=provider)
    if damage == "userinfo_url":
        assert raw["sft_exclusions"] == ["privacy_nonexact"]
        assert "privacy_redactions" in raw["completeness"]["gaps"]
    else:
        assert raw["response"] == provider
        assert raw["sft_exclusions"] == ["target_shape_or_completion_unsupported"]
    assert candidate is None


def test_opaque_chat_citation_projection_keeps_all_visible_citation_fields(tmp_path):
    provider = response()
    expected = json.loads(json.dumps(provider["choices"][0]["message"]["annotations"]))
    provider["choices"][0]["message"]["annotations"][0]["url_citation"]["encrypted_content"] = "opaque"
    raw, candidate = export(tmp_path, provider=provider)
    assert candidate["target"]["annotations"] == expected
    assert candidate["source"]["response_projection"] == {
        "opaque_reasoning_omitted": True, "exact_provider_replay": False}
    assert raw["completeness"]["dimensions"]["privacy"] is False
    assert "privacy_redactions" in raw["completeness"]["gaps"]


@pytest.mark.parametrize("damage", ["undeclared", "unknown_tool", "unknown_control", "bad_mode", "bool_limit"])
def test_internal_chat_annotations_require_declared_valid_search_controls(tmp_path, damage):
    body = request()
    if damage == "undeclared": body.pop("tools"); body.pop("max_tool_calls")
    elif damage == "unknown_tool": body["tools"][0]["type"] = "unknown_tool"
    elif damage == "unknown_control": body["tools"][0]["parameters"]["unreviewed"] = True
    elif damage == "bad_mode": body["tools"][0]["parameters"]["mode"] = "unknown"
    else: body["max_tool_calls"] = True
    raw, candidate = export(tmp_path, body=body)
    assert raw["request"] == body and candidate is None


def test_nonexact_secret_annotation_remains_excluded_and_not_duplicated(tmp_path):
    provider = response()
    provider["choices"][0]["message"]["annotations"][0]["url_citation"]["content"] = "test-provider-secret"
    raw, candidate = export(tmp_path, provider=provider, secrets=("test-provider-secret",))
    assert candidate is None and raw["sft_exclusions"] == ["privacy_nonexact"]
    assert "privacy_redactions" in raw["completeness"]["gaps"]
    assert "test-provider-secret" not in json.dumps(raw)
    assert raw["research_evidence"]["chat_url_citations"]["annotations_with_content"] == 1


def test_default_miner_gate_still_rejects_the_same_nested_annotations():
    with pytest.raises(ValueError, match="assistant_shape_unsupported"):
        training_steps._target(request(), response(), "openrouter.chat", allow_reasoning_history=True)
