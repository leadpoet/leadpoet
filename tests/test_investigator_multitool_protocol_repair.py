"""A rejected research batch gets at most the existing serialization repair."""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.linkedin_company_size import MALFORMED_RESPONSE_FAILURE_REASON
from tests.test_company_evidence_investigator import _attribute_finding, _finding

URL = "https://acme.example/product"
QUOTE = "Acme supplies subscription software to customers."


def _call(name, arguments, call_id="one"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def _batch():
    return [_call("fetch_page", {"url": URL}, "fetch-one"),
            _call("fetch_page", {"url": URL + "/details"}, "fetch-two"),
            _call("search_web", {"query": "Acme subscription software"}, "search")]


def _submit():
    return _call("submit_findings", {"findings": [
        _attribute_finding("VERIFIED", url=URL, quote=QUOTE),
    ]}, "submit")


def _run(monkeypatch, responses, *, expired=False, turns=None, targets=("required_attribute",)):
    requests, searches, fetches = [], [], []
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now

    async def post(_session, _url, *, headers, payload):
        requests.append(copy.deepcopy(payload))
        assert len(requests) <= len(responses), "unbudgeted model retry"
        response = copy.deepcopy(responses[len(requests) - 1])
        if expired:
            clock.now = investigator.ADMISSION_DEADLINE_SECONDS
        return 200, response

    async def search(_session, query, **_kwargs):
        searches.append(query)
        return {"results": [], "notice": "discovery_only_not_evidence"}

    async def fetch(_session, url):
        fetches.append(url)
        raise AssertionError("rejected batch must not fetch")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "time", clock)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    if turns is not None:
        monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", turns)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=targets, requested_attribute="Supplies subscription software",
        requested_employee_buckets=["51-200"],
        prior_observations={"submitted_source_urls": [URL]},
        prefetched_pages={URL: {"text": QUOTE, "final_url": URL}},
    ))
    return result, requests, searches, fetches


def _response(calls):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "tool_calls": calls,
    }}]}


@pytest.mark.parametrize("names", [
    ("fetch_page", "fetch_page", "search_web"),  # retained 57 / cache replay 67
    ("fetch_page", "fetch_page", "fetch_page"),  # retained 63
])
def test_valid_research_batch_is_not_executed_and_changes_corrective_request(monkeypatch, names):
    batch = [_call(name, {"url": URL} if name == "fetch_page" else {"query": "Acme"}, str(i))
             for i, name in enumerate(names)]
    result, requests, searches, fetches = _run(monkeypatch, [
        _response(batch), _response([_submit()]),
    ])
    assert result["_completed_submit"] is True
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert len(requests) == 2
    assert searches == fetches == []
    assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
    assert requests[0] != requests[1]
    assert requests[1]["messages"][-1]["role"] == "user"
    assert "None was executed" in requests[1]["messages"][-1]["content"]
    assert requests[1]["parallel_tool_calls"] is False
    assert requests[1]["tool_choice"] == requests[0]["tool_choice"]


def test_valid_single_findings_still_completes_without_repair(monkeypatch):
    result, requests, searches, fetches = _run(monkeypatch, [_response([_submit()])])
    assert result["_completed_submit"] is True
    assert len(requests) == 1
    assert searches == fetches == []


@pytest.mark.parametrize("mode", ["repeated", "truncated_after", "truncated_before", "malformed_submit_after"])
def test_multitool_and_serialization_share_one_repair(monkeypatch, mode):
    empty = {"choices": [{"finish_reason": "length", "message": {"content": None}}],
             "usage": {"completion_tokens": 3000}}
    multi = _response(_batch())
    malformed_submit = _response([_call("submit_findings", {"findings": []})])
    responses = {"repeated": [multi, multi], "truncated_after": [multi, empty],
                 "truncated_before": [empty, multi],
                 "malformed_submit_after": [multi, malformed_submit]}[mode]
    result, requests, searches, fetches = _run(monkeypatch, responses)
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert not result.get("_completed_submit")
    assert len(requests) == 2
    assert searches == fetches == []


@pytest.mark.parametrize("mode", ["duplicate_id", "empty_id", "unknown_tool", "invalid_json", "wrong_args", "non_object_args", "invalid_url", "overlong_query", "non_call", "submit_batch", "zero_calls", "ambiguous_choices"])
def test_invalid_batch_remains_fail_closed_without_repair(monkeypatch, mode):
    batch = _batch()
    if mode == "duplicate_id":
        batch[1]["id"] = batch[0]["id"]
    elif mode == "empty_id":
        batch[0]["id"] = ""
    elif mode == "unknown_tool":
        batch[0]["function"]["name"] = "execute_code"
    elif mode == "invalid_json":
        batch[0]["function"]["arguments"] = '{"url":'
    elif mode == "wrong_args":
        batch[0]["function"]["arguments"] = '{"query":"wrong"}'
    elif mode == "non_object_args":
        batch[0]["function"]["arguments"] = '[]'
    elif mode == "invalid_url":
        batch[0]["function"]["arguments"] = '{"url":"file:///etc/passwd"}'
    elif mode == "overlong_query":
        batch[-1]["function"]["arguments"] = json.dumps({"query": "x" * 501})
    elif mode == "non_call":
        batch[0] = None
    elif mode == "submit_batch":
        batch = [_submit(), dict(_submit(), id="other-submit")]
    elif mode == "zero_calls":
        batch = []
    response = _response(batch)
    if mode == "ambiguous_choices":
        response["choices"] *= 2
    result, requests, searches, fetches = _run(monkeypatch, [response])
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert len(requests) == 1
    assert searches == fetches == []


@pytest.mark.parametrize("expired,turns", [(True, None), (False, 1)])
def test_multitool_repair_cannot_extend_time_or_turn_budget(monkeypatch, expired, turns):
    result, requests, searches, fetches = _run(
        monkeypatch, [_response(_batch())], expired=expired, turns=turns,
    )
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert len(requests) == 1
    assert searches == fetches == []


def test_multitool_cannot_change_a_forced_tool():
    assert investigator._valid_multi_research_reply(_batch(), required_tool="")
    assert not investigator._valid_multi_research_reply(_batch(), required_tool="search_web")
    assert not investigator._valid_multi_research_reply(_batch(), required_tool="submit_findings")


@pytest.mark.parametrize("query,valid", [("Acme latest financing", True),
                                        ("acme.example latest financing", True),
                                        ("Unrelated company financing", False)])
def test_multitool_preserves_forced_stage_search_company_binding(query, valid):
    batch = [_call("search_web", {"query": query}, str(i)) for i in range(2)]
    assert investigator._valid_multi_research_reply(
        batch, required_tool="search_web", required_search_locators=("acme", "acme.example"),
    ) is valid


def test_wrong_forced_tool_batch_gets_no_repair_or_research(monkeypatch):
    unproven = _call("submit_findings", {"findings": [_finding(
        "headcount", status="UNPROVEN", observed_value=None,
        evidence_url="", evidence_quote="",
    )]})
    result, requests, searches, fetches = _run(monkeypatch, [
        _response([unproven]), _response(_batch()),
    ], targets=("headcount",))
    assert requests[1]["tool_choice"] == {
        "type": "function", "function": {"name": "search_web"},
    }
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert len(requests) == 2
    assert searches == fetches == []
