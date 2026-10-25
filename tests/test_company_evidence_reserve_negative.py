"""A time-reserved research denial cannot complete an unproven judgment."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.product_coverage_fixtures import coverage, trusted_plan
from tests.test_company_evidence_submit_retry import URL, QUOTE, _finding, _tool_response


CRITERION = "Enrollment software for universities"


@pytest.mark.parametrize("final_status", ["UNPROVEN", "VERIFIED"])
@pytest.mark.parametrize("denied_tool", ["search_web", "fetch_page"])
def test_reserve_denied_research_only_interrupts_unproven_final(
    monkeypatch, final_status, denied_tool,
):
    now = [0.0]
    trusted_plan(CRITERION)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(investigator, "MAX_SEARCH_CALLS", 2)
    monkeypatch.setattr(
        investigator, "time", SimpleNamespace(monotonic=lambda: now[0]),
    )
    requests = []

    async def post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            if denied_tool == "search_web":
                now[0] = (
                    investigator.ADMISSION_DEADLINE_SECONDS
                    - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS + 1
                )
            name, arguments = "search_web", {"query": "Acme enrollment proof"}
        elif turn == 2 and denied_tool == "fetch_page":
            now[0] = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS + 1
            )
            name, arguments = "fetch_page", {"url": "https://acme.example/news"}
        else:
            finding = _finding(product_coverage=coverage(
                CRITERION, URL, QUOTE,
                unproven=("p0",) if final_status == "UNPROVEN" else (),
            ))
            if final_status == "UNPROVEN":
                finding.update(
                    status="UNPROVEN", observed_value="", observed_industry="",
                    observed_subindustry="", activity_role="unresolved",
                    evidence_url="", evidence_quote="", reason="No proof loaded.",
                )
            name, arguments = "submit_findings", {"findings": [finding]}
        return 200, _tool_response(name, json.dumps(arguments))

    async def search(_session, _query, *, key):
        del key
        now[0] = 50.0
        return {"results": []}

    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",), requested_industry="Education",
        requested_subindustry="Higher education services",
        requested_product_service=CRITERION, positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={
            "normalized_name": "acme", "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))

    assert len(requests) == (2 if denied_tool == "search_web" else 3)
    assert json.loads(requests[-1]["messages"][-1]["content"])["error"] == (
        "judgment_time_reserved"
    )
    assert result["usage"]["search_calls"] == int(denied_tool == "fetch_page")
    assert result["usage"]["fetch_calls"] == 0
    assert result["claims"]["industry"]["status"] == final_status
    if final_status == "UNPROVEN":
        assert result["_completed_submit"] is False
        assert result["failure_reason"] == (
            investigator.ADMISSION_BUDGET_INTERRUPTED_FAILURE_REASON
        )
    else:
        assert result["_completed_submit"] is True
        assert result["failure_reason"] == ""
