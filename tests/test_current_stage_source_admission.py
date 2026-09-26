"""Narrow current-stage source and entity-binding regressions."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer


VISTA_URL = "https://www.vistaequitypartners.com/about/team/nick-prickel/"
VISTA_QUOTE = (
    "Nick Prickel joined Vista Equity Partners in 2012 and is a member of the "
    "private equity Flagship team. Nick has been actively involved in numerous "
    "Vista investments, including active majority investments in Acumatica, "
    "Applause, Avalara, Gainsight, KnowBe4, Model N, Smartbear, Smartsheet and "
    "Solera, and other investments sold to sponsors and strategic buyers, "
    "including Aderant, Aptean, BigMachines, Bonterra, Granicus and Jamf."
)


def _validate_stage(name, stage, url, quote, *, first_party_domain=""):
    return investigator._validated_findings(
        {"findings": [{
            "target": "stage",
            "status": "VERIFIED",
            "observed_value": stage,
            "evidence_url": url,
            "evidence_quote": quote,
        }]},
        targets=("stage",),
        fetched_pages={url: quote},
        first_party_domains={first_party_domain} if first_party_domain else set(),
        identity_names={name.casefold().replace(" ", "")},
    )["stage"]


def _company(name, website):
    return CompanyOutput(
        company_name=name,
        company_website=website,
        company_linkedin="",
        industry="Software",
        employee_count="51-200",
        country="United States",
        intent_signals=[{
            "description": "hiring",
            "source": "company website",
            "url": f"{website.rstrip('/')}/careers",
            "date": "2026-09-01",
            "snippet": f"{name} is hiring.",
        }],
    )


def test_vista_active_majority_list_proves_only_targets_before_sold_boundary():
    assert _validate_stage(
        "Acumatica", "Private Equity", VISTA_URL, VISTA_QUOTE,
    )["status"] == "VERIFIED"
    for exited_name in ("Aptean", "Jamf"):
        assert _validate_stage(
            exited_name, "Private Equity", VISTA_URL, VISTA_QUOTE,
        )["status"] == "UNPROVEN"


@pytest.mark.parametrize("quote", [
    "Vista Equity Partners, a private equity firm, lists Acumatica in its portfolio.",
    "Vista Equity Partners made a minority investment in Acumatica.",
    "Vista Equity Partners made a growth investment in Acumatica.",
    "Vista Equity Partners proposed an acquisition of Acumatica.",
    "Acumatica was formerly a majority investment of Vista Equity Partners, a private equity firm.",
])
def test_portfolio_minority_growth_proposal_and_historical_pe_are_unproven(quote):
    assert _validate_stage(
        "Acumatica", "Private Equity", VISTA_URL, quote,
    )["status"] == "UNPROVEN"


def test_bare_secondary_public_claim_is_rejected_in_both_admission_paths():
    url = "https://mergr.com/company/unanet"
    quote = "Unanet is publicly traded."
    assert _validate_stage("Unanet", "Public", url, quote)["status"] == "UNPROVEN"

    verdict = {
        "observed_company_stage": "Public",
        "observed_company_name": "Unanet",
        "observed_company_website": "https://unanet.com",
        "stage_matches": True,
        "stage_evidence_url": url,
        "stage_evidence_quote": quote,
    }
    assert lead_scorer._decision_from_observed_stage(
        verdict,
        "public",
        company=_company("Unanet", "https://unanet.com"),
    ) == lead_scorer.COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize("quote", [
    "Unanet raised funding from Onex, whose shares are listed on the Toronto Stock Exchange.",
    "Unanet received financing from ParentCo (NASDAQ: PCO).",
    "Unanet's parent is listed on the NYSE.",
    "Unanet shares are listed on its website.",
])
def test_target_cannot_inherit_parent_or_investor_listing(quote):
    assert _validate_stage(
        "Unanet", "Public", "https://profiles.example/unanet", quote,
    )["status"] == "UNPROVEN"


def test_current_issuer_first_party_public_evidence_is_preserved():
    url = "https://www.tenable.com/investor-relations/company-profile"
    bare_quote = "Tenable is publicly traded."
    market_quote = "Tenable Holdings, Inc. (NASDAQ: TENB) provides exposure management."
    assert _validate_stage(
        "Tenable", "Public", url, bare_quote, first_party_domain="tenable.com",
    )["status"] == "VERIFIED"
    assert _validate_stage(
        "Tenable", "Public", url, market_quote, first_party_domain="tenable.com",
    )["status"] == "VERIFIED"
    verdict = {
        "observed_company_stage": "Public",
        "observed_company_name": "Tenable",
        "observed_company_website": "https://tenable.com",
        "stage_matches": True,
        "stage_evidence_url": url,
        "stage_evidence_quote": bare_quote,
    }
    assert lead_scorer._decision_from_observed_stage(
        verdict,
        "public",
        company=_company("Tenable", "https://tenable.com"),
    ) == lead_scorer.COMPANY_FIT_MATCH


def test_secondary_current_exchange_locator_remains_accepted_for_issuer():
    url = "https://profiles.example/tenable"
    quote = "Tenable Holdings, Inc. (NASDAQ: TENB) provides exposure management."
    assert _validate_stage("Tenable", "Public", url, quote)["status"] == "VERIFIED"


def test_first_party_historical_ipo_only_does_not_establish_current_listing():
    url = "https://tenable.com/news/2018-initial-public-offering"
    quote = "In July 2018, Tenable completed its initial public offering."
    assert _validate_stage(
        "Tenable", "Public", url, quote, first_party_domain="tenable.com",
    )["status"] == "UNPROVEN"


def test_sec_registered_equity_path_remains_accepted():
    url = "https://www.sec.gov/Archives/edgar/data/1777319/form10-q.htm"
    quote = (
        "CISO GLOBAL, INC. (Exact name of registrant as specified in its charter) "
        "Securities registered pursuant to Section 12(b) of the Act: Title of "
        "each class Trading Symbol(s) Name of each exchange on which registered "
        "Common Stock CISO The Nasdaq Stock Market LLC"
    )
    assert _validate_stage("CISO Global", "Public", url, quote)["status"] == "VERIFIED"


def test_private_equity_request_gets_existing_bounded_stage_discovery(monkeypatch):
    requests = []
    search = AsyncMock(return_value={"results": [{"url": VISTA_URL}]})

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps({"findings": [{
                    "target": "stage",
                    "status": "UNPROVEN",
                    "observed_value": None,
                    "reason": "no source fetched",
                }]}),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", AsyncMock())

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Acumatica", "website": "https://acumatica.com",
        },
        targets=("stage",),
        requested_stage="Private Equity",
    ))

    expected_query = (
        "Acumatica acumatica.com current owner completed acquisition "
        "majority private equity"
    )
    search.assert_awaited_once()
    assert search.await_args.args[1] == expected_query
    assert result["usage"]["search_calls"] == 1
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["server_current_stage_discovery"]["query"] == expected_query
    assert document["investigation_limits"]["remaining_search_calls"] == 1


def test_pe_rejected_prefetch_uses_remaining_search_and_fetch(monkeypatch):
    weak_url = "https://www.bigtime.net/about-us/"
    strong_url = "https://www.bigtime.net/news/current-owner"
    weak_quote = "BigTime Software is a private equity-funded company."
    strong_quote = (
        "BigTime Software is owned by Vista Equity Partners, a private equity firm."
    )
    requests = []
    searches = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "submit_findings", {"findings": [{
                "target": "stage", "status": "VERIFIED",
                "observed_value": "Private Equity",
                "evidence_url": weak_url, "evidence_quote": weak_quote,
            }]}
        elif turn == 2:
            name, arguments = "search_web", {
                "query": "BigTime current controlling private equity owner",
            }
        elif turn == 3:
            name, arguments = "fetch_page", {"url": strong_url}
        else:
            name, arguments = "submit_findings", {"findings": [{
                "target": "stage", "status": "VERIFIED",
                "observed_value": "Private Equity",
                "evidence_url": strong_url, "evidence_quote": strong_quote,
            }]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": [{"url": strong_url}]}

    fetch = AsyncMock(return_value={
        "ok": True, "url": strong_url, "final_url": strong_url,
        "text": strong_quote,
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "BigTime", "website": "https://bigtime.net"},
        targets=("stage",),
        requested_stage="Private Equity",
        prior_observations={"submitted_source_urls": [weak_url]},
        prefetched_pages={
            weak_url: {"final_url": weak_url, "text": weak_quote},
        },
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["usage"] == {
        "reasoning_turns": 4, "search_calls": 2, "fetch_calls": 1,
    }
    assert len(searches) == investigator.MAX_SEARCH_CALLS
    fetch.assert_awaited_once()
    assert requests[1]["tool_choice"] == {
        "type": "function", "function": {"name": "search_web"},
    }


def test_pe_rejected_prefetch_does_not_exceed_full_fetch_budget(monkeypatch):
    weak_url = "https://bigtime.example/about"
    other_urls = (
        "https://bigtime.example/platform",
        "https://bigtime.example/company",
    )
    weak_quote = "BigTime Software is a private equity-funded company."
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            finding = {
                "target": "stage", "status": "VERIFIED",
                "observed_value": "Private Equity",
                "evidence_url": weak_url, "evidence_quote": weak_quote,
            }
        else:
            finding = {
                "target": "stage", "status": "UNPROVEN",
                "observed_value": None, "reason": "no controlling owner proof",
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}", "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps({"findings": [finding]}),
            },
        }]}}]}

    search = AsyncMock(return_value={"results": []})
    fetch = AsyncMock()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    pages = {
        weak_url: {"final_url": weak_url, "text": weak_quote},
        **{
            url: {"final_url": url, "text": "BigTime Software page."}
            for url in other_urls
        },
    }
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "BigTime Software", "website": "https://bigtime.example",
        },
        targets=("stage",),
        requested_stage="Private Equity",
        prior_observations={
            "submitted_source_urls": [weak_url, *other_urls],
        },
        prefetched_pages=pages,
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"] == {
        "reasoning_turns": 2, "search_calls": 1, "fetch_calls": 0,
    }
    search.assert_awaited_once()
    fetch.assert_not_awaited()
    assert requests[1]["tool_choice"] == "required"
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["investigation_limits"]["remaining_fetch_calls"] == 0
