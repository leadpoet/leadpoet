"""Ordinary reserve-denied research does not erase a valid final judgment."""

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
@pytest.mark.parametrize("other_unproven", [False, True])
def test_ordinary_reserve_denial_preserves_complete_final(
    monkeypatch, final_status, denied_tool, other_unproven,
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
            findings = [finding]
            if other_unproven:
                findings.append(_finding(
                    target="headcount", status="UNPROVEN",
                    observed_value="", observed_industry="",
                    observed_subindustry="", activity_role="unresolved",
                    evidence_url="", evidence_quote="",
                    reason="No public headcount proof.",
                ))
            name, arguments = "submit_findings", {"findings": findings}
        return 200, _tool_response(name, json.dumps(arguments))

    async def search(_session, _query, *, key, max_results=5):
        del key
        now[0] = 50.0
        return {"results": []}

    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=(("industry", "headcount") if other_unproven else ("industry",)),
        requested_employee_buckets=("11-50",) if other_unproven else (),
        requested_industry="Education",
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
    if other_unproven:
        assert result["claims"]["headcount"]["status"] == "UNPROVEN"
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""


@pytest.mark.parametrize("deny_forced_fetch", [True, False])
def test_industry_followup_fetch_denial_is_bound_to_pending_obligation(
    monkeypatch, deny_forced_fetch,
):
    now = [0.0]
    home = "https://acme.example/about"
    followup = "https://acme.example/software"
    quote = "Acme uses software supplied by Example Systems."
    requests = []
    fetches = []

    async def post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": home}
        elif turn == 2:
            name, arguments = "submit_findings", {"findings": [_finding(
                status="CONTRADICTED", observed_value="Customer use",
                observed_industry="Other activity", activity_role="customer_user",
                evidence_url=home, evidence_quote=quote,
            )]}
        elif turn == 3:
            assert payload["tool_choice"]["function"]["name"] == "search_web"
            name, arguments = "search_web", {"query": "Acme software supplier"}
        elif turn == 4:
            assert payload["tool_choice"]["function"]["name"] == "fetch_page"
            if deny_forced_fetch:
                now[0] = (
                    investigator.ADMISSION_DEADLINE_SECONDS
                    - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS + 1
                )
            name, arguments = "fetch_page", {"url": followup}
        elif turn == 5 and not deny_forced_fetch:
            assert payload["tool_choice"] == "required"
            now[0] = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS + 1
            )
            name, arguments = "fetch_page", {"url": "https://acme.example/other"}
        else:
            assert payload["tool_choice"]["function"]["name"] == "submit_findings"
            name, arguments = "submit_findings", {"findings": [_finding(
                status="UNPROVEN", observed_value="", observed_industry="",
                observed_subindustry="", activity_role="unresolved",
                evidence_url="", evidence_quote="", reason="Proof not loaded.",
            )]}
        return 200, _tool_response(name, json.dumps(arguments), call_id=f"call-{turn}")

    async def search(_session, _query, *, key, max_results=5):
        del key
        return {"results": [{"url": followup}]}

    async def fetch(_session, url):
        fetches.append(url)
        return {"ok": True, "url": url, "final_url": url, "text": quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(investigator, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",), requested_industry="Software",
        verified_homepage_identity={
            "normalized_name": "acme", "registrable_dns_domain": "acme.example",
        },
    ))

    assert len(requests) == (5 if deny_forced_fetch else 6)
    assert fetches == ([home] if deny_forced_fetch else [home, followup])
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["_completed_submit"] is not deny_forced_fetch
    assert result["failure_reason"] == (
        investigator.ADMISSION_BUDGET_INTERRUPTED_FAILURE_REASON
        if deny_forced_fetch else ""
    )
