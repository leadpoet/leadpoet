"""Search summaries count retained text, not privacy placeholders."""
from copy import deepcopy
import json

import pytest

from lab_arena import internal_capture
from lab_arena.service import LocalObjectStore
from tests.lab_arena.internal_capture_responses_test import request, response


@pytest.mark.parametrize("location", ["sources", "results"])
@pytest.mark.parametrize("field", ["text", "content", "snippet"])
@pytest.mark.parametrize("value,count,status", [
    ("Retained source excerpt.", 1, "exposed"),
    ("[REDACTED]", 0, "redacted"),
    (" \n[REDACTED]\t", 0, "redacted"),
    ("", 0, "not_exposed"),
    (" \n\t", 0, "not_exposed"),
])
def test_responses_text_summary_distinguishes_retained_redacted_and_absent(location, field, value, count, status):
    provider = {"output": [{"type": "web_search_call", "status": "completed",
                            "action": {location: [{field: value}]}}]}
    before = deepcopy(provider)
    evidence = internal_capture._search_evidence(provider)
    assert evidence["events_with_result_text"] == count
    assert evidence["all_search_events_have_result_text"] is bool(count)
    assert evidence["events"][0]["results_with_text"] == count
    assert evidence["events"][0]["result_text_status"] == status
    assert provider == before


@pytest.mark.parametrize("value,count", [
    ("Retained source excerpt.", 1), ("[REDACTED]", 0),
    (" \n[REDACTED]\t", 0), ("", 0), (" \n\t", 0),
])
def test_chat_text_summary_uses_the_same_placeholder_boundary(value, count):
    provider = {"choices": [{"message": {"annotations": [{"type": "url_citation",
                "url_citation": {"url": "https://example.com/source", "content": value}}]}}]}
    before = deepcopy(provider)
    citations = internal_capture._search_evidence(provider)["chat_url_citations"]
    assert citations["references"] == 1
    assert citations["annotations_with_content"] == count
    assert citations["content_characters"] == (len(value) if count else 0)
    assert (citations["items"][0]["content_path"] is not None) is bool(count)
    assert provider == before


def test_mixed_events_keep_real_text_counts_and_original_output_order():
    provider = {"output": [
        {"type": "reasoning"},
        {"type": "web_search_call", "action": {"sources": [{"text": "[REDACTED]"}]}},
        {"type": "reasoning"},
        {"type": "web_search_call", "action": {"sources": [{"url": "https://example.com"}]}},
        {"type": "reasoning"},
        {"type": "web_search_call", "action": {"results": [
            {"text": "[REDACTED]", "snippet": "Real retained snippet."},
            {"content": " \t[REDACTED] "}]}},
    ], "usage": {"server_tool_use_details": {"web_search_requests": 3}}}
    before = deepcopy(provider)
    evidence = internal_capture._search_evidence(provider)
    assert [r["output_index"] for r in evidence["events"]] == [1, 3, 5]
    assert [r["result_text_status"] for r in evidence["events"]] == ["redacted", "not_exposed", "exposed"]
    assert [r["results_with_text"] for r in evidence["events"]] == [0, 0, 1]
    assert evidence["events_with_result_text"] == 1
    assert evidence["all_search_events_have_result_text"] is False
    assert evidence["provider_reported_web_search_requests"]["count"] == 3
    assert provider == before


def test_privacy_removed_text_stays_distinct_from_provider_nonexposure(tmp_path):
    objects = LocalObjectStore(tmp_path)
    provider = response()
    provider["output"][1]["action"]["sources"][0]["text"] = "fixture-private-token"
    provider["usage"].update(input_tokens=121, output_tokens=33,
                             output_tokens_details={"reasoning_tokens": 0}, cost=0.0002)
    before = deepcopy(provider)
    call = internal_capture.InternalCapture(objects).start(
        purpose="icp_generation", context={}, request=request(), api="openrouter.responses",
        secrets=("fixture-private-token",))
    call.response(body=json.dumps(provider).encode(), status=200)
    call.finish(outcome={"status": "fixture"}, cost_microusd=200)
    raw, candidate = internal_capture.read_call(objects, call.prefix + "request.json")
    assert raw["response"]["output"][1]["action"]["sources"][0]["text"] == "[REDACTED]"
    assert raw["research_evidence"]["events"][0]["result_text_status"] == "redacted"
    assert raw["research_evidence"]["events"][1]["result_text_status"] == "not_exposed"
    assert raw["research_evidence"]["events_with_result_text"] == 0
    assert raw["events"]["response"]["capture"]["redacted"] is True
    assert raw["response"]["usage"] == provider["usage"]
    assert [r["id"] for r in raw["response"]["output"]] == [r["id"] for r in provider["output"]]
    assert candidate is None and raw["sft_exclusions"] == ["privacy_nonexact"]
    assert provider == before
