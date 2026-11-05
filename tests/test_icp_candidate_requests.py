"""Candidate probes use real prompts and retain native Responses evidence."""

import asyncio
import copy
import json

import httpx
import pytest

from gateway.tasks import icp_generator as generator
from lab_arena.internal_capture import read_call
from tests.test_icp_generator_dynamic_count import _provider_rows
from tests.test_icp_internal_capture import _capture


CANDIDATE = {"model": "openai/gpt-oss-120b", "provider": "deepinfra", "effort": "high"}


def _native(content, *, count=2):
    return {"id": "resp-candidate", "status": "completed", "usage": {
        "cost": "0.012", "server_tool_use_details": {"web_search_requests": count},
    }, "output": [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Plan searches"}]},
        {"type": "openrouter:web_search", "status": "completed", "action": {
            "query": "first evidence", "results": [{"url": "https://example.com/first", "content": "first source"}],
        }},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Check another source"}]},
        {"type": "openrouter:web_search", "status": "completed", "action": {
            "query": "second evidence", "results": [{"url": "https://example.com/second", "content": "second source"}],
        }},
        {"type": "message", "role": "assistant", "status": "completed", "content": [
            {"type": "output_text", "text": content},
        ]},
    ]}


@pytest.mark.parametrize("effort", ["low", "high"])
def test_offline_production_requests_keep_defaults_and_candidates_pin_route(monkeypatch, effort):
    def no_io(*args, **kwargs):
        pytest.fail("offline preparation performed I/O")

    monkeypatch.setattr(generator.httpx, "AsyncClient", no_io)
    monkeypatch.setattr(generator.httpx, "post", no_io)
    monkeypatch.setattr(generator, "_IcpCapture", no_io)
    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "")
    baseline = asyncio.run(generator.prepare_icp_probe_request(
        "icp_generation", set_id=20261105, total_icps=10,
    ))
    assert baseline["api"] == "openrouter.chat"
    assert baseline["path"] == "/chat/completions"
    before = copy.deepcopy(baseline["request"])
    assert before["model"] == "anthropic/claude-sonnet-5.5"
    assert before["tools"][0]["parameters"] == {"engine": "native", "max_uses": 10}
    api, candidate = generator.prepare_icp_candidate_request(before, {**CANDIDATE, "effort": effort})
    assert before == baseline["request"]
    assert api == "openrouter.responses"
    assert candidate["input"] == before["messages"]
    assert candidate["reasoning"] == {"effort": effort, "context": "all_turns"}
    assert candidate["include"] == ["reasoning.encrypted_content"]
    assert candidate["provider"] == {
        "only": ["deepinfra"], "allow_fallbacks": False,
        "require_parameters": True, "zdr": True, "data_collection": "deny",
    }
    assert candidate["tools"] == [{"type": "openrouter:web_search", "parameters": {
        "engine": "exa", "max_uses": 10,
    }}]
    assert candidate["max_tool_calls"] == before["max_tool_calls"] == 10
    assert candidate["max_output_tokens"] == before["max_tokens"]
    assert candidate["transforms"] == []
    assert candidate["plugins"] == [{"id": "context-compression", "enabled": False}]
    assert candidate["stream"] is candidate["store"] is False
    exclusion = asyncio.run(generator.prepare_icp_probe_request(
        "icp_exclusion", icp={"industry": "Software"},
    ))
    assert exclusion["request"]["model"] == "perplexity/sonar-pro"
    assert exclusion["request"]["temperature"] == 0.0
    assert "tools" not in exclusion["request"]
    selected = asyncio.run(generator.prepare_icp_probe_request(
        "icp_exclusion", icp={"industry": "Software"}, candidate=CANDIDATE,
    ))
    assert selected["api"] == "openrouter.responses"
    assert selected["request"]["max_output_tokens"] == 8192
    assert "temperature" not in selected["request"]


def test_candidate_generation_uses_normalizer_and_saves_all_exposed_native_output(monkeypatch, tmp_path):
    spy = _capture(monkeypatch, tmp_path)
    observed = {}
    dumps = json.dumps

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, headers, json):
            observed.update(url=url, request=json)
            assert headers["X-OpenRouter-Metadata"] == "enabled"
            slots = json["input"][0]["content"].split(
                "ORDERED INDUSTRY SLOTS (one ICP per slot, in this order; repeated names are intentional):\n", 1,
            )[1].split("\n", 1)[0].split(", ")
            observed["response"] = _native(dumps({"icps": _provider_rows(slots)}))
            return httpx.Response(200, json=observed["response"])

    monkeypatch.setattr(generator.httpx, "AsyncClient", Client)
    result = asyncio.run(generator.generate_icps_with_openrouter(
        20261105, total_icps=10, api_key="test-key", candidate=CANDIDATE,
    ))
    assert len(result[0]) == 10
    assert result[2] == generator.compute_icp_set_hash(result[0])
    assert observed["url"].endswith("/responses")
    call, = spy.calls
    assert call["request"]["api"] == "openrouter.responses"
    raw, candidate = read_call(spy.core.objects, call["prefix"] + "request.json")
    assert raw["request"] == observed["request"]
    assert raw["response"] == observed["response"]
    assert len(raw["response"]["output"]) == 5
    assert raw["outcome"]["normalized_icps"] == result[0]
    assert raw["outcome"]["search_capture"] == {
        "reported_search_requests": 2, "exposed_search_events": 2,
        "completed_search_events": 2, "search_event_count_matches": True,
        "all_internal_provider_turns_verified": False,
    }


def test_candidate_exclusion_saves_searches_before_final_text_projection(monkeypatch, tmp_path):
    spy = _capture(monkeypatch, tmp_path)
    response = _native('[{"domain":"https://www.excluded.example/path","name":"Excluded"}]')
    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs["json"]))
        return httpx.Response(200, json=response)

    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(generator.httpx, "post", post)
    assert generator.generate_exclusions_for_icp({"industry": "Software"}, candidate=CANDIDATE) == ["excluded.example"]
    assert requests[0][0].endswith("/responses")
    raw, candidate = read_call(spy.core.objects, spy.calls[0]["prefix"] + "request.json")
    assert raw["response"] == response
    assert raw["outcome"]["normalized_output"] == ["excluded.example"]
    assert raw["cost_microusd"] == 12000


def test_search_count_mismatch_and_unknown_usage_are_explicit():
    mismatch = generator.icp_search_capture_status(_native("[]", count=3))
    assert mismatch["reported_search_requests"] == 3
    assert mismatch["search_event_count_matches"] is False
    missing = generator.icp_search_capture_status({"output": [], "usage": {}})
    assert missing["reported_search_requests"] is None
    assert missing["search_event_count_matches"] is False
    assert missing["all_internal_provider_turns_verified"] is False
    assert generator.icp_search_capture_status({"output": {}, "usage": []}) == missing


@pytest.mark.parametrize("change", [
    {"status": "incomplete"}, {"error": {"code": "provider_failure"}},
    {"output": [{"type": "function_call", "name": "unfinished"}]},
    {"output": [{"type": "web_search_call", "status": "in_progress"}]},
    {"output": [{"type": "message", "role": "assistant", "content": [{"type": "refusal"}]}]},
])
def test_candidate_incomplete_or_unfinished_output_never_enters_normalizer(change):
    with pytest.raises(ValueError):
        generator.project_icp_candidate_response({**_native("[]"), **change}, api="openrouter.responses")


@pytest.mark.parametrize("change", [
    {"provider": ""}, {"model": "router/auto", "fallback": True},
    {"effort": "none"}, {"max_output_tokens": True}, {"max_output_tokens": 0},
])
def test_candidate_controls_reject_ambiguous_or_unbounded_routes(change):
    with pytest.raises(ValueError):
        generator.prepare_icp_candidate_request({"messages": []}, {**CANDIDATE, **change})
