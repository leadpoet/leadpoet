"""A late admitted judgment keeps only the grader's existing final correction."""

from __future__ import annotations

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_judgment_reserve import (
    QUOTE, URL, _Clock, _finding, _response, _run_industry,
)


def _run_correction(monkeypatch, *, final, cross_after=8, positive=False):
    clock = _Clock()
    clock.now = 0.0
    first_clock_read = True

    def monotonic():
        nonlocal first_clock_read
        if first_clock_read:
            first_clock_read = False
            return 0.0
        # The first seven ordinary replies reuse a loaded page. The eighth
        # turn is the existing forced judgment; no fresh research is needed.
        return clock.now

    monkeypatch.setattr(clock, "monotonic", monotonic)
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        turn = len(requests)
        if cross_after and turn == cross_after:
            # The ordinary request was admitted before the deadline. Its
            # response settles late, with no provider cancellation or retry.
            clock.now = 180.0
        if turn < investigator.MAX_REASONING_TURNS:
            return 200, _response("fetch_page", {"url": URL}, call=turn)
        if turn == investigator.MAX_REASONING_TURNS:
            return 200, _response("submit_findings", {
                "findings": [_finding(url="", quote="")],
            }, call=turn)
        assert turn == investigator.MAX_REASONING_TURNS + 1
        if final == "provider_failure":
            return 503, {"error": "provider unavailable"}
        if final == "malformed_arguments":
            return 200, _response("submit_findings", '{"findings":[', call=turn)
        if final == "research_tool":
            return 200, _response("fetch_page", {"url": "https://acme.example/other"}, call=turn)
        finding = (
            _finding(status="UNPROVEN") if final == "unproven"
            else _finding(quote="Acme manufactures aircraft.") if final == "fabricated_quote"
            else _finding(url="", quote="") if final == "missing_citation"
            else _finding()
        )
        return 200, _response("submit_findings", {"findings": [finding]}, call=turn)

    async def no_research(*_args, **_kwargs):
        raise AssertionError("the existing final correction cannot perform research")

    monkeypatch.setattr(investigator, "_search_web", no_research)
    monkeypatch.setattr(investigator, "_fetch_page", no_research)
    result = _run_industry(
        monkeypatch, post, clock, positive_semantic_review=positive,
    )
    return result, requests


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("final,status", [
    ("verified", "VERIFIED"), ("unproven", "UNPROVEN"),
    ("fabricated_quote", "UNPROVEN"), ("missing_citation", "UNPROVEN"),
])
def test_late_eighth_judgment_keeps_existing_ninth_correction_and_validation(
    monkeypatch, positive, final, status,
):
    # The same model replies finish under the original fast-clock grading
    # path. Only admission timing changes; every request and result is equal.
    fast, fast_requests = _run_correction(
        monkeypatch, final=final, cross_after=None, positive=positive,
    )
    late, late_requests = _run_correction(
        monkeypatch, final=final, positive=positive,
    )
    assert investigator.MAX_REASONING_TURNS == 8
    assert len(late_requests) == len(fast_requests) == 9
    assert late_requests == fast_requests
    assert late == fast
    rejected = final in {"fabricated_quote", "missing_citation"}
    assert late["_completed_submit"] is (not rejected)
    assert late["failure_reason"] == ("malformed_response" if rejected else "")
    assert late["usage"]["reasoning_turns"] == 9
    assert late["usage"]["search_calls"] == late["usage"]["fetch_calls"] == 0
    assert late["claims"]["industry"]["status"] == status
    if final == "verified":
        assert late["claims"]["industry"]["evidence_quote"] == QUOTE
    elif final == "fabricated_quote":
        assert late["claims"]["industry"]["evidence_quote"] == ""
        assert late["claims"]["industry"]["reason"] == (
            "submitted quote was not present in fetched source"
        )
    assert [tool["function"]["name"] for tool in late_requests[-1]["tools"]] == [
        "submit_findings",
    ]
    assert late_requests[-1]["tool_choice"]["function"]["name"] == "submit_findings"


@pytest.mark.parametrize("final,reason", [
    ("provider_failure", "provider_error"),
    ("malformed_arguments", "malformed_response"),
    ("research_tool", "malformed_response"),
])
def test_final_correction_fault_does_not_create_a_tenth_turn(monkeypatch, final, reason):
    result, requests = _run_correction(monkeypatch, final=final)
    assert len(requests) == 9
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == reason


def test_late_research_settlement_allows_only_judgment_and_its_final_correction(monkeypatch):
    # An admitted research reply may settle late. Loaded evidence gets one
    # submit-only judgment, then only its existing pending final correction.
    result, requests = _run_correction(monkeypatch, final="verified", cross_after=7)
    assert len(requests) == 9
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["usage"]["reasoning_turns"] == 9
    assert result["usage"]["fetch_calls"] == result["usage"]["search_calls"] == 0
    for request in requests[7:]:
        assert request["tool_choice"]["function"]["name"] == "submit_findings"
        assert [tool["function"]["name"] for tool in request["tools"]] == ["submit_findings"]
