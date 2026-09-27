"""Bounded re-review for fetched issuer proof left UNPROVEN by the model."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer


TENB_URL = (
    "https://www.tenable.com/press-releases/"
    "tenable-appoints-dino-dimarino-as-chief-revenue-officer"
)
TENB_QUOTE = (
    "Tenable Holdings, Inc. (NASDAQ: TENB), the exposure management company, "
    "today announced the appointment of Dino DiMarino as Chief Revenue Officer."
)


def _finding(*, status="UNPROVEN", url="", quote="", value=None, reason=""):
    return {
        "target": "stage",
        "status": status,
        "observed_value": value,
        "evidence_url": url,
        "evidence_quote": quote,
        "reason": reason or ("current stage unresolved" if status == "UNPROVEN" else "verified"),
    }


def _tool_response(turn, finding):
    return 200, {"choices": [{"message": {"tool_calls": [{
        "id": f"call-{turn}",
        "type": "function",
        "function": {
            "name": "submit_findings",
            "arguments": json.dumps({"findings": [finding]}),
        },
    }]}}]}


def _prefetched_request(
    monkeypatch,
    *,
    company_name,
    company_url,
    evidence_url,
    evidence_text,
    findings,
    search=None,
):
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        finding = findings[min(len(requests) - 1, len(findings) - 1)]
        return _tool_response(len(requests), finding)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(
        investigator,
        "_search_web",
        search or AsyncMock(return_value={"results": []}),
    )
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": company_name, "website": company_url},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={"submitted_source_urls": [evidence_url]},
        prefetched_pages={evidence_url: {
            "final_url": evidence_url,
            "text": evidence_text,
        }},
    ))
    return result, requests


def test_tenable_optional_filing_failure_gets_one_bounded_rereview(monkeypatch):
    filing_url = "https://www.sec.gov/Archives/edgar/data/1660280/tenb-20251231.htm"
    product_url = "https://www.tenable.com/products/exposure-management"
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "fetch_page", {"url": filing_url}
        elif turn == 2:
            name, arguments = "fetch_page", {"url": product_url}
        elif turn == 3:
            name, arguments = "submit_findings", {"findings": [_finding(
                reason=(
                    "The fetched issuer announcement identifies NASDAQ: TENB, "
                    "but the optional current filing could not be read."
                ),
            )]}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(
                status="VERIFIED",
                value="Public",
                url=TENB_URL,
                quote=TENB_QUOTE,
                reason=(
                    "The fetched current issuer announcement identifies Tenable "
                    "Holdings, Inc. as NASDAQ: TENB, and discovery exposed no "
                    "completed superseding event."
                ),
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_fetch(_session, url, **_kwargs):
        if url == TENB_URL:
            return {"ok": True, "url": url, "final_url": url, "text": TENB_QUOTE}
        if url == filing_url:
            return {"ok": False, "error": "http_403"}
        return {
            "ok": True,
            "url": url,
            "final_url": url,
            "text": "Tenable provides exposure management software.",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(
        investigator,
        "_search_web",
        AsyncMock(return_value={"results": [{"url": filing_url}]}),
    )

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={
            "submitted_source_urls": [TENB_URL],
            "submitted_source_hints": [{"url": TENB_URL, "text": TENB_QUOTE}],
        },
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["evidence_url"] == TENB_URL
    assert result["usage"]["reasoning_turns"] == 4
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 3
    correction = json.loads(requests[3]["messages"][-1]["content"])
    assert correction["error"] == "fetched_public_stage_evidence_not_adjudicated"
    assert correction["issuer_bound_market_source_urls"] == [TENB_URL]
    assert "failed optional fetch" in correction["instruction"]
    assert requests[3]["tool_choice"] == {
        "type": "function", "function": {"name": "submit_findings"},
    }


def test_other_issuer_gets_the_same_general_rereview(monkeypatch):
    url = "https://www.crowdstrike.com/news/current-results"
    quote = "CrowdStrike Holdings, Inc. (NASDAQ: CRWD) announced current results."
    result, requests = _prefetched_request(
        monkeypatch,
        company_name="CrowdStrike",
        company_url="https://crowdstrike.com",
        evidence_url=url,
        evidence_text=quote,
        findings=[
            _finding(reason="optional filing unavailable"),
            _finding(status="VERIFIED", value="Public", url=url, quote=quote),
        ],
    )

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert len(requests) == 2


def test_solarwinds_material_delisting_conflict_stays_unproven(monkeypatch):
    url = "https://www.solarwinds.com/news/transaction"
    text = (
        "SolarWinds Corporation (NYSE: SWI) announced results. "
        "Turn/River Capital completed the acquisition of SolarWinds Corporation, "
        "and SolarWinds common stock ceased trading on the New York Stock Exchange."
    )
    unproven = _finding(reason="later completed acquisition and delisting conflict")
    result, requests = _prefetched_request(
        monkeypatch,
        company_name="SolarWinds",
        company_url="https://solarwinds.com",
        evidence_url=url,
        evidence_text=text,
        findings=[unproven, unproven],
    )

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert len(requests) == 2
    correction = json.loads(requests[1]["messages"][-1]["content"])
    assert "material conflict" in correction["instruction"]


@pytest.mark.parametrize("quote", [
    (
        "In July 2018, Tenable Holdings, Inc. (NASDAQ: TENB) completed its "
        "initial public offering."
    ),
    (
        "Tenable Holdings, Inc. (NASDAQ: TENB) completed its initial public "
        "offering in July 2018."
    ),
])
def test_stale_ipo_ticker_does_not_trigger_public_rereview(monkeypatch, quote):
    url = "https://tenable.com/news/2018-initial-public-offering"
    result, requests = _prefetched_request(
        monkeypatch,
        company_name="Tenable",
        company_url="https://tenable.com",
        evidence_url=url,
        evidence_text=quote,
        findings=[_finding(reason="historical IPO only")],
    )

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert len(requests) == 1


def _validated_public_finding(quote):
    url = "https://www.tenable.com/news/issuer-update"
    return investigator._validated_findings(
        {"findings": [_finding(
            status="VERIFIED",
            value="Public",
            url=url,
            quote=quote,
        )]},
        targets=("stage",),
        fetched_pages={url: quote},
        fetched_final_urls={url: url},
        first_party_domains={"tenable.com"},
        identity_names={"tenable", "tenableholdings"},
        identity_anchor={
            "submitted_name": "Tenable",
            "submitted_domain": "tenable.com",
        },
    )["stage"]


@pytest.mark.parametrize("quote", [
    (
        "In July 2018, Tenable Holdings, Inc. (NASDAQ: TENB) completed its "
        "initial public offering."
    ),
    (
        "Tenable Holdings, Inc. (NASDAQ: TENB) completed its initial public "
        "offering in July 2018."
    ),
])
def test_ipo_completion_only_ticker_is_not_current_public_proof(quote):
    assert not lead_scorer._public_quote_has_bound_market_locator(
        quote, ("tenable", "tenableholdings"),
    )
    assert _validated_public_finding(quote)["status"] == "UNPROVEN"


@pytest.mark.parametrize("quote", [
    (
        "Tenable Holdings, Inc. (NASDAQ: TENB), today announced quarterly "
        "results. In July 2018, Tenable completed its initial public offering."
    ),
    (
        "In July 2018, Tenable completed its initial public offering. "
        "Tenable Holdings, Inc. (NASDAQ: TENB), today announced quarterly "
        "results."
    ),
])
def test_independent_current_issuer_row_survives_historical_ipo_elsewhere(quote):
    assert lead_scorer._public_quote_has_bound_market_locator(
        quote, ("tenable", "tenableholdings"),
    )
    assert _validated_public_finding(quote)["status"] == "VERIFIED"


@pytest.mark.parametrize("current_statement", [
    "Tenable common stock is currently listed on Nasdaq under ticker TENB",
    "Tenable common stock currently trades on Nasdaq under ticker TENB",
])
def test_same_clause_explicit_current_listing_survives_historical_ipo(
    current_statement,
):
    quote = (
        "Tenable Holdings, Inc. (NASDAQ: TENB) completed its initial public "
        f"offering in July 2018 and {current_statement}."
    )

    assert lead_scorer._public_quote_has_bound_market_locator(
        quote, ("tenable", "tenableholdings"),
    )
    assert _validated_public_finding(quote)["status"] == "VERIFIED"


def test_wrong_issuer_current_ticker_does_not_rescue_tenable_ipo():
    quote = (
        "Tenable Holdings, Inc. (NASDAQ: TENB) completed its initial public "
        "offering in July 2018. Other Holdings, Inc. (NASDAQ: OTHR), today "
        "announced quarterly results."
    )

    assert not lead_scorer._public_quote_has_bound_market_locator(
        quote, ("tenable", "tenableholdings"),
    )
    assert _validated_public_finding(quote)["status"] == "UNPROVEN"


@pytest.mark.parametrize("quote", [
    (
        "Tenable Holdings, Inc. (NASDAQ: TENB) announced quarterly results "
        "on September 27, 2026."
    ),
    "Tenable common stock is currently listed on Nasdaq under ticker TENB.",
])
def test_current_issuer_market_statements_remain_public_proof(quote):
    assert lead_scorer._public_quote_has_bound_market_locator(
        quote, ("tenable", "tenableholdings"),
    )
    assert _validated_public_finding(quote)["status"] == "VERIFIED"


def test_wrong_issuer_market_locator_does_not_trigger_rereview(monkeypatch):
    url = "https://tenable.com/news/partner-results"
    result, requests = _prefetched_request(
        monkeypatch,
        company_name="Tenable",
        company_url="https://tenable.com",
        evidence_url=url,
        evidence_text="ParentCo Holdings, Inc. (NASDAQ: PCO) announced current results.",
        findings=[_finding(reason="ticker belongs to another issuer")],
    )

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert len(requests) == 1


def test_failed_current_status_search_does_not_trigger_rereview(monkeypatch):
    search = AsyncMock(side_effect=RuntimeError("search unavailable"))
    result, requests = _prefetched_request(
        monkeypatch,
        company_name="Tenable",
        company_url="https://tenable.com",
        evidence_url=TENB_URL,
        evidence_text=TENB_QUOTE,
        findings=[_finding(reason="current-status search failed")],
        search=search,
    )

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert len(requests) == 1


def test_no_bound_market_evidence_does_not_trigger_rereview(monkeypatch):
    url = "https://tenable.com/products"
    result, requests = _prefetched_request(
        monkeypatch,
        company_name="Tenable",
        company_url="https://tenable.com",
        evidence_url=url,
        evidence_text="Tenable provides exposure management software.",
        findings=[_finding(reason="no listing evidence")],
    )

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert len(requests) == 1


def test_no_available_reasoning_turn_does_not_add_a_call(monkeypatch):
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 1)
    result, requests = _prefetched_request(
        monkeypatch,
        company_name="Tenable",
        company_url="https://tenable.com",
        evidence_url=TENB_URL,
        evidence_text=TENB_QUOTE,
        findings=[_finding(reason="optional filing unavailable")],
    )

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["reasoning_turns"] == 1
    assert len(requests) == 1


def test_repeated_unproven_gets_only_one_rereview(monkeypatch):
    unproven = _finding(reason="current state remains unresolved")
    result, requests = _prefetched_request(
        monkeypatch,
        company_name="Tenable",
        company_url="https://tenable.com",
        evidence_url=TENB_URL,
        evidence_text=TENB_QUOTE,
        findings=[unproven, unproven, unproven],
    )

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert len(requests) == 2
    correction_count = sum(
        "fetched_public_stage_evidence_not_adjudicated" in str(request["messages"])
        for request in requests
    )
    assert correction_count == 1
