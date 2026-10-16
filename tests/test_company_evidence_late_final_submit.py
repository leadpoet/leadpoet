from __future__ import annotations

import asyncio

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_judgment_reserve import (
    QUOTE, URL, _Clock, _finding, _response,
)


def _investigate(monkeypatch, post, clock, *, loaded=True, positive=False):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "time", clock)
    return asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",),
        positive_semantic_review=positive,
        requested_industry="Education",
        requested_subindustry="Higher education services",
        requested_product_service="Enrollment software for universities",
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={
            "normalized_name": "acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}} if loaded else {},
    ))


@pytest.mark.parametrize("late_tool", ["fetch_page", "search_web"])
@pytest.mark.parametrize("elapsed", [110.0, 122.5, 180.0])
@pytest.mark.parametrize("positive", [False, True])
def test_late_reasoning_gets_one_submit_only_turn_without_new_research(
    monkeypatch, late_tool, elapsed, positive,
):
    clock = _Clock()
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            # A successful ordinary reasoning call returns a research tool
            # after its admission window. Its selected tool must not dispatch.
            clock.now = elapsed
            arguments = ({"url": "https://acme.example/other"}
                         if late_tool == "fetch_page" else {"query": "Acme"})
            return 200, _response(late_tool, arguments)
        assert len(requests) == 2
        assert payload["tool_choice"]["function"]["name"] == "submit_findings"
        assert [tool["function"]["name"] for tool in payload["tools"]] == [
            "submit_findings",
        ]
        return 200, _response("submit_findings", {"findings": [_finding()]}, call=2)

    async def fail_research(*_args, **_kwargs):
        raise AssertionError("no new research may dispatch after admission closes")

    monkeypatch.setattr(investigator, "_fetch_page", fail_research)
    monkeypatch.setattr(investigator, "_search_web", fail_research)
    result = _investigate(monkeypatch, post, clock, positive=positive)

    assert len(requests) == 2
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["fetch_calls"] == result["usage"]["search_calls"] == 0


def test_admitted_successful_late_fetch_is_judged(monkeypatch):
    clock = _Clock()
    requests = []
    fetches = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            return 200, _response("fetch_page", {"url": URL})
        assert len(requests) == 2
        assert payload["tool_choice"]["function"]["name"] == "submit_findings"
        return 200, _response("submit_findings", {"findings": [_finding()]}, call=2)

    async def fetch(_session, url, **_kwargs):
        fetches.append(url)
        clock.now = 111.0
        return {"ok": True, "url": url, "final_url": url, "text": QUOTE}

    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = _investigate(monkeypatch, post, clock, loaded=False)

    assert fetches == [URL]
    assert len(requests) == 2
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["fetch_calls"] == 1


def test_already_admitted_valid_late_submit_needs_no_extra_turn(monkeypatch):
    clock = _Clock()
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        clock.now = 180.0
        return 200, _response("submit_findings", {"findings": [_finding()]})

    result = _investigate(monkeypatch, post, clock)

    assert len(requests) == 1
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"


def test_admitted_priority_prefetch_can_finish_with_one_late_submit(monkeypatch):
    clock = _Clock()
    requests = []
    fetches = []

    async def fetch(_session, url, **_kwargs):
        fetches.append(url)
        clock.now = 111.0
        return {"ok": True, "url": url, "final_url": url, "text": QUOTE}

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        assert payload["tool_choice"]["function"]["name"] == "submit_findings"
        assert [tool["function"]["name"] for tool in payload["tools"]] == [
            "submit_findings",
        ]
        return 200, _response("submit_findings", {"findings": [_finding()]})

    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = _investigate(monkeypatch, post, clock, loaded=False, positive=True)

    assert fetches == [URL]
    assert len(requests) == 1
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"


def test_late_final_submit_cannot_clear_failed_mandatory_stage_discovery(monkeypatch):
    clock = _Clock()
    requests = []

    async def search(*_args, **_kwargs):
        raise RuntimeError("search unavailable")

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            clock.now = 122.5
            return 200, _response("fetch_page", {"url": URL})
        assert len(requests) == 2
        finding = dict(_finding(status="UNPROVEN"), target="stage")
        return 200, _response("submit_findings", {"findings": [finding]}, call=2)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "time", clock)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",), requested_stage="Public",
        prior_observations={"submitted_source_urls": [URL]},
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))

    assert len(requests) == 2
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == "provider_error"
    assert result["claims"]["stage"]["status"] == "UNPROVEN"


@pytest.mark.parametrize("failure", ["empty_evidence", "provider_failure"])
def test_late_failure_cannot_create_a_final_judgment(monkeypatch, failure):
    clock = _Clock()
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        clock.now = 122.5
        if failure == "provider_failure":
            return 503, {"error": "unavailable"}
        return 200, _response("fetch_page", {"url": URL})

    result = _investigate(
        monkeypatch, post, clock, loaded=failure != "empty_evidence",
    )

    assert len(requests) == 1
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == (
        "provider_error" if failure == "provider_failure"
        else "admission_budget_interrupted"
    )


@pytest.mark.parametrize("bad_final", [
    "fabricated_quote", "research_tool", "provider_failure",
])
def test_late_final_turn_cannot_relax_evidence_or_start_another_turn(
    monkeypatch, bad_final,
):
    clock = _Clock()
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            clock.now = 122.5
            return 200, _response("fetch_page", {"url": URL})
        assert len(requests) == 2
        if bad_final == "provider_failure":
            return 503, {"error": "unavailable"}
        if bad_final == "research_tool":
            return 200, _response("search_web", {"query": "Acme"}, call=2)
        return 200, _response("submit_findings", {"findings": [
            _finding(quote="A fabricated statement absent from the loaded source."),
        ]}, call=2)

    result = _investigate(monkeypatch, post, clock)

    assert len(requests) == 2
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] in {
        "admission_budget_interrupted", "malformed_response", "provider_error",
    }
    assert result.get("claims", {}).get("industry", {}).get("status") != "VERIFIED"


@pytest.mark.parametrize("elapsed,expected_calls", [(122.5, 3), (140.0, 2)])
def test_late_submit_uses_only_existing_serialization_repair(
    monkeypatch, elapsed, expected_calls,
):
    clock = _Clock()
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            clock.now = elapsed
            return 200, _response("fetch_page", {"url": URL})
        if len(requests) == 2:
            response = _response("submit_findings", '{"findings":[')
            response["choices"][0]["finish_reason"] = "length"
            return 200, response
        assert len(requests) == 3
        assert payload["tool_choice"]["function"]["name"] == "submit_findings"
        return 200, _response("submit_findings", {"findings": [_finding()]}, call=3)

    result = _investigate(monkeypatch, post, clock, positive=True)

    assert len(requests) == expected_calls
    assert (result.get("_completed_submit") is True) is (expected_calls == 3)
    if expected_calls == 2:
        assert result["failure_reason"] == "malformed_response"
