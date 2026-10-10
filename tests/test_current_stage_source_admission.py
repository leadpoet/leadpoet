"""Narrow current-stage source and entity-binding regressions."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import company_evidence_investigator as investigator


def _core_usage(usage):
    return {key: usage[key] for key in ("reasoning_turns", "search_calls", "fetch_calls")}
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
        identity_names={re.sub(r"[^a-z0-9]+", "", name.casefold())},
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


TENB_NASDAQ_URL = (
    "https://www.nasdaq.com/press-release/"
    "tenable-appoints-dino-dimarino-chief-revenue-officer-2026-03-12"
)
TENB_FIRST_PARTY_URL = (
    "https://www.tenable.com/press-releases/"
    "tenable-appoints-dino-dimarino-as-chief-revenue-officer"
)
TENB_QUOTE = (
    "Tenable Holdings, Inc. (NASDAQ: TENB), the exposure management company, "
    "today announced the appointment of Dino DiMarino as Chief Revenue Officer."
)


def _select_public_source(*, urls, hints, pages=None, disputes=()):
    return investigator._public_stage_submitted_source_to_prefetch(
        submitted_source_urls=urls,
        stage_dispute_urls=disputes,
        submitted_source_hints=hints,
        first_party_domains={"tenable.com"},
        identity_names={"tenable", "tenableholdingsinc"},
        fetched_pages=pages or {},
    )


@pytest.mark.parametrize(
    ("url", "hint", "identity_names", "generic_page"),
    [
        (
            "https://www.nasdaq.com/press-release/crowdstrike-appoints-"
            "bartley-richardson-chief-ai-and-autonomous-systems-officer-2026",
            "“CrowdStrike (NASDAQ: CRWD) today announced the appointment of "
            "Dr. Bartley Richardson as Chief AI and Autonomous Systems Officer. "
            "Richardson now leads CrowdStrike’s AI strategy”",
            {"crowdstrike"},
            "CrowdStrike (NASDAQ: CRWD) reported quarterly results.",
        ),
        (
            "https://www.nasdaq.com/press-release/rapid7-appoints-wael-"
            "mohamed-chief-executive-officer-corey-thomas-become-executive",
            "“Rapid7, Inc. (NASDAQ: RPD), a global leader in AI-powered managed "
            "cybersecurity operations, today announced a leadership transition "
            "in which board member Wael Mohamed will assume the role of Chief "
            "Executive Officer”",
            {"rapid7", "rapid7inc"},
            "Rapid7, Inc. (NASDAQ: RPD) reported quarterly results.",
        ),
        (
            "https://www.sentinelone.com/press/sentinelone-appoints-sonalee-"
            "parekh-as-chief-financial-officer/",
            "“SentinelOne (NYSE: S), the leader in AI-native cybersecurity, "
            "today announced the appointment of Sonalee Parekh as Chief "
            "Financial Officer, effective March 24, 2026.”",
            {"sentinelone"},
            "SentinelOne (NYSE: S) reported quarterly results.",
        ),
        (
            "https://www.tenable.com/press-releases/tenable-appoints-dino-"
            "dimarino-as-chief-revenue-officer",
            "“Tenable® Holdings, Inc. (NASDAQ: TENB), the exposure management "
            "company, today announced the appointment of Dino DiMarino as Chief "
            "Revenue Officer (CRO).”",
            {"tenable", "tenableholdingsinc"},
            "Tenable Holdings, Inc. (NASDAQ: TENB) reported quarterly results.",
        ),
    ],
)
def test_wrapped_frozen_issuer_hint_outranks_generic_fetched_stage_page(
    url, hint, identity_names, generic_page,
):
    generic_url = "https://profiles.example/company"

    assert investigator._public_stage_submitted_source_to_prefetch(
        submitted_source_urls=[generic_url, url],
        stage_dispute_urls=[],
        submitted_source_hints={url: [hint]},
        first_party_domains={"sentinelone.com", "tenable.com"},
        identity_names=identity_names,
        fetched_pages={generic_url: generic_page},
    ) == url


@pytest.mark.parametrize(
    "hint",
    [
        '"Acme Holdings, Inc. (NASDAQ: ACME) announced current results.',
        "Acme Holdings, Inc. (NASDAQ: ACME) announced current results.",
    ],
)
def test_straight_wrapped_and_unwrapped_issuer_hints_remain_selectable(hint):
    url = "https://exchange.example/releases/acme-results"

    assert investigator._public_stage_submitted_source_to_prefetch(
        submitted_source_urls=[url],
        stage_dispute_urls=[],
        submitted_source_hints={url: [hint]},
        first_party_domains=set(),
        identity_names={"acme", "acmeholdingsinc"},
        fetched_pages={},
    ) == url


@pytest.mark.parametrize(
    "hint",
    [
        "“Other Holdings, Inc. (NASDAQ: OTHR) announced current results.”",
        "Acme Holdings plans to list on NASDAQ under ticker ACME.",
    ],
)
def test_different_issuer_or_no_listing_proof_hint_is_not_selected(hint):
    url = "https://exchange.example/releases/unbound-result"

    assert investigator._public_stage_submitted_source_to_prefetch(
        submitted_source_urls=[url],
        stage_dispute_urls=[],
        submitted_source_hints={url: [hint]},
        first_party_domains=set(),
        identity_names={"acme", "acmeholdingsinc"},
        fetched_pages={},
    ) == ""


def test_forged_submitted_hint_cannot_bypass_exact_fetched_quote_admission():
    url = "https://acme.example/releases/results"
    forged_hint = "Acme Holdings, Inc. (NASDAQ: ACME) announced results."

    finding = investigator._validated_findings(
        {"findings": [{
            "target": "stage",
            "status": "VERIFIED",
            "observed_value": "Public",
            "evidence_url": url,
            "evidence_quote": forged_hint,
        }]},
        targets=("stage",),
        fetched_pages={url: "Acme Holdings announced a new product."},
        first_party_domains={"acme.example"},
        identity_names={"acme", "acmeholdingsinc"},
    )["stage"]

    assert finding["status"] == "UNPROVEN"
    assert finding["evidence_url"] == ""
    assert finding["evidence_quote"] == ""


def test_tenable_nasdaq_hint_is_selected_despite_inherited_industry_page():
    industry_url = "https://www.tenable.com/products"
    assert _select_public_source(
        urls=[industry_url, TENB_NASDAQ_URL],
        hints={TENB_NASDAQ_URL: [TENB_QUOTE]},
        pages={industry_url: "Tenable provides exposure management software."},
    ) == TENB_NASDAQ_URL


def test_strong_first_party_public_hint_precedes_secondary_hint():
    assert _select_public_source(
        urls=[TENB_NASDAQ_URL, TENB_FIRST_PARTY_URL],
        hints={
            TENB_NASDAQ_URL: [TENB_QUOTE],
            TENB_FIRST_PARTY_URL: [TENB_QUOTE],
        },
    ) == TENB_FIRST_PARTY_URL


def test_actual_prefetched_public_page_avoids_duplicate_fetch():
    assert _select_public_source(
        urls=[TENB_NASDAQ_URL],
        hints={TENB_NASDAQ_URL: [TENB_QUOTE]},
        pages={TENB_NASDAQ_URL: TENB_QUOTE},
    ) == ""


@pytest.mark.parametrize("hint", [
    "Unanet is publicly traded.",
    "Unanet received financing from ParentCo (NASDAQ: PCO).",
    "Unanet raised funding from Onex, whose shares are listed on the Toronto Stock Exchange.",
])
def test_unanet_weak_or_other_issuer_hint_is_not_selected(hint):
    url = "https://profiles.example/unanet"
    assert investigator._public_stage_submitted_source_to_prefetch(
        submitted_source_urls=[url],
        stage_dispute_urls=[],
        submitted_source_hints={url: [hint]},
        first_party_domains={"unanet.com"},
        identity_names={"unanet"},
        fetched_pages={},
    ) == ""


def test_completed_take_private_dispute_precedes_old_ticker_hint():
    dispute_url = "https://news.example/solarwinds-take-private-completed"
    ticker_url = "https://markets.example/solarwinds-old-listing"
    assert investigator._public_stage_submitted_source_to_prefetch(
        submitted_source_urls=[ticker_url, dispute_url],
        stage_dispute_urls=[dispute_url],
        submitted_source_hints={
            ticker_url: ["SolarWinds Corporation (NYSE: SWI) announced results."],
        },
        first_party_domains={"solarwinds.com"},
        identity_names={"solarwinds", "solarwindscorporation"},
        fetched_pages={},
    ) == dispute_url


def test_solarwinds_old_ticker_match_reopens_public_stage_without_submitted_conflict():
    result = SimpleNamespace(details={
        "dimension_decisions": {
            "stage": lead_scorer.COMPANY_FIT_MATCH,
            "industry": lead_scorer.COMPANY_FIT_MATCH,
        },
        "provider_observations": {},
    })

    assert lead_scorer._targeted_company_investigation_dimensions(
        result,
        icp_stage="Public",
        employee_size_conflict=False,
        company=_company("SolarWinds", "https://solarwinds.com"),
    ) == ("stage",)


def test_public_request_gets_bounded_current_stage_discovery(monkeypatch):
    requests = []
    search = AsyncMock(return_value={"results": []})

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
                    "reason": "no current listing or supersession source fetched",
                }]}),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "SolarWinds", "website": "https://solarwinds.com",
        },
        targets=("stage",),
        requested_stage="Public",
    ))

    expected_query = (
        "SolarWinds solarwinds.com current public listing "
        "completed take-private acquisition delisting"
    )
    search.assert_awaited_once()
    assert search.await_args.args[1] == expected_query
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["search_calls"] == 1
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["server_current_stage_discovery"]["query"] == expected_query
    assert document["server_current_stage_discovery"]["ok"] is True
    assert document["investigation_limits"]["remaining_search_calls"] == 1


def test_matching_public_requires_successful_current_status_search(monkeypatch):
    async def fake_post(_session, _url, *, headers, payload):
        del headers, payload
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps({"findings": [{
                    "target": "stage",
                    "status": "VERIFIED",
                    "observed_value": "Public",
                    "evidence_url": TENB_FIRST_PARTY_URL,
                    "evidence_quote": TENB_QUOTE,
                }]}),
            },
        }]}}]}

    search = AsyncMock(side_effect=RuntimeError("search unavailable"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={
            "submitted_source_urls": [TENB_FIRST_PARTY_URL],
        },
        prefetched_pages={
            TENB_FIRST_PARTY_URL: {
                "final_url": TENB_FIRST_PARTY_URL,
                "text": TENB_QUOTE,
            },
        },
    ))

    search.assert_awaited_once()
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["claims"]["stage"]["reason"] == (
        "current public stage was not established after a successful "
        "company-bound current-status discovery"
    )
    assert result["_validated_stage_finding"] == {}


def test_current_tenable_public_proof_survives_successful_current_search(monkeypatch):
    async def fake_post(_session, _url, *, headers, payload):
        del headers, payload
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "submit_findings",
                "arguments": json.dumps({"findings": [{
                    "target": "stage",
                    "status": "VERIFIED",
                    "observed_value": "Public",
                    "evidence_url": TENB_FIRST_PARTY_URL,
                    "evidence_quote": TENB_QUOTE,
                }]}),
            },
        }]}}]}

    search = AsyncMock(return_value={"results": []})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={
            "submitted_source_urls": [TENB_FIRST_PARTY_URL],
        },
        prefetched_pages={
            TENB_FIRST_PARTY_URL: {
                "final_url": TENB_FIRST_PARTY_URL,
                "text": TENB_QUOTE,
            },
        },
    ))

    search.assert_awaited_once()
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["observed_value"] == "Public"
    assert result["_validated_stage_finding"] == result["claims"]["stage"]


def test_completed_solarwinds_take_private_is_a_deterministic_contradiction(
    monkeypatch,
):
    source_url = (
        "https://www.solarwinds.com/company/newsroom/press-releases/"
        "turnriver-completes-acquisition-of-solarwinds"
    )
    quote = (
        "Turn/River Capital has completed the acquisition of SolarWinds "
        "Corporation. With the closing of the transaction, SolarWinds common "
        "stock has ceased trading on the New York Stock Exchange."
    )
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            name = "fetch_page"
            arguments = {"url": source_url}
        else:
            name = "submit_findings"
            arguments = {"findings": [{
                "target": "stage",
                "status": "CONTRADICTED",
                "observed_value": "Acquired",
                "evidence_url": source_url,
                "evidence_quote": quote,
            }]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    search = AsyncMock(return_value={"results": [{"url": source_url}]})
    fetch = AsyncMock(return_value={
        "ok": True,
        "url": source_url,
        "final_url": source_url,
        "text": quote,
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "SolarWinds", "website": "https://solarwinds.com",
        },
        targets=("stage",),
        requested_stage="Public",
    ))

    search.assert_awaited_once()
    fetch.assert_awaited_once()
    assert result["claims"]["stage"]["status"] == "CONTRADICTED"
    assert result["claims"]["stage"]["observed_value"] == "Acquired"
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 2,
        "search_calls": 1,
        "fetch_calls": 1,
    }


@pytest.mark.parametrize("quote", [
    "Turn/River Capital proposed an acquisition of SolarWinds Corporation.",
    "Turn/River Capital completed the acquisition of Another Company.",
])
def test_proposal_or_wrong_entity_cannot_contradict_solarwinds_public_stage(quote):
    finding = _validate_stage(
        "SolarWinds",
        "Acquired",
        "https://news.example/transaction",
        quote,
    )
    assert finding["status"] == "UNPROVEN"


def test_tenable_nasdaq_hint_fetches_within_budget_and_exact_quote_admits(monkeypatch):
    industry_url = "https://www.tenable.com/products"
    requests = []

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
                    "status": "VERIFIED",
                    "observed_value": "Public",
                    "evidence_url": TENB_NASDAQ_URL,
                    "evidence_quote": TENB_QUOTE,
                }]}),
            },
        }]}}]}

    fetch = AsyncMock(return_value={
        "ok": True,
        "url": TENB_NASDAQ_URL,
        "final_url": TENB_NASDAQ_URL,
        "text": TENB_QUOTE,
    })
    search = AsyncMock(return_value={"results": [{"url": TENB_NASDAQ_URL}]})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={
            "submitted_source_urls": [industry_url, TENB_NASDAQ_URL],
            "submitted_source_hints": [{
                "url": TENB_NASDAQ_URL,
                "text": TENB_QUOTE,
            }],
        },
        prefetched_pages={
            industry_url: {
                "final_url": industry_url,
                "text": "Tenable provides exposure management software.",
            },
        },
        verified_homepage_identity={
            "normalized_name": "Tenable",
            "registrable_dns_domain": "tenable.com",
        },
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 1,
        "search_calls": 1,
        "fetch_calls": 1,
    }
    fetch.assert_awaited_once()
    search.assert_awaited_once()
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["investigation_limits"]["prefetched_pages"] == 1
    assert document["investigation_limits"]["server_prefetch_fetch_calls"] == 1
    assert document["investigation_limits"]["remaining_fetch_calls"] == 2
    assert document["investigation_limits"]["remaining_search_calls"] == 1


def test_diagnostic_pages_leave_existing_locator_room_to_recover(monkeypatch):
    bad_urls = (
        "https://example.com/first",
        "https://example.com/second",
    )
    good_url = TENB_NASDAQ_URL
    requests = []
    fetched = []
    actions = [
        ("fetch_page", {"url": bad_urls[0]}),
        ("fetch_page", {"url": bad_urls[1]}),
        ("fetch_page", {"url": good_url}),
        ("submit_findings", {"findings": [{
            "target": "stage", "status": "VERIFIED", "observed_value": "Public",
            "evidence_url": good_url, "evidence_quote": TENB_QUOTE,
        }]}),
    ]

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        name, arguments = actions[len(requests) - 1]
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_bounded(_session, url, **_kwargs):
        fetched.append(url)
        body = (
            "Provider account capacity details redacted."
            if url in bad_urls else f"<main>{TENB_QUOTE}</main>"
        )
        return 200, url, body

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.delenv("SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.delenv("QUALIFICATION_SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_bounded_html", fake_bounded)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={
        "results": [{"url": url} for url in (*bad_urls, good_url)],
    }))

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com"},
        targets=("stage",), requested_stage="Public",
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["evidence_url"] == good_url
    assert fetched == [*bad_urls, good_url]
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 4, "search_calls": 1, "fetch_calls": 3,
    }
    assert result["usage"]["total_loaded_pages"] == 1
    assert [item["error_class"] for item in result["usage"]["fetch_outcomes"]] == [
        "provider_diagnostic_body", "provider_diagnostic_body", "",
    ]
    assert "Provider account capacity details redacted." not in json.dumps(requests)
    assert TENB_QUOTE in json.dumps(requests[-1])


def test_tenable_hint_cannot_admit_quote_absent_from_fetched_source():
    finding = _validate_stage(
        "Tenable",
        "Public",
        TENB_NASDAQ_URL,
        TENB_QUOTE,
    )
    assert finding["status"] == "VERIFIED"
    rejected = investigator._validated_findings(
        {"findings": [{
            "target": "stage",
            "status": "VERIFIED",
            "observed_value": "Public",
            "evidence_url": TENB_NASDAQ_URL,
            "evidence_quote": TENB_QUOTE,
        }]},
        targets=("stage",),
        fetched_pages={
            TENB_NASDAQ_URL: "Tenable announced a leadership appointment."
        },
        first_party_domains={"tenable.com"},
        identity_names={"tenable", "tenableholdingsinc"},
    )["stage"]
    assert rejected["status"] == "UNPROVEN"
    assert rejected["reason"] == "submitted quote was not present in fetched source"


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
    hinted_url = "https://markets.example/acumatica"
    requests = []
    search = AsyncMock(return_value={"results": [{"url": VISTA_URL}]})
    fetch = AsyncMock()

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
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Acumatica", "website": "https://acumatica.com",
        },
        targets=("stage",),
        requested_stage="Private Equity",
        prior_observations={
            "submitted_source_urls": [hinted_url],
            "submitted_source_hints": [{
                "url": hinted_url,
                "text": "Acumatica Holdings, Inc. (NASDAQ: ACME) announced results.",
            }],
        },
    ))

    expected_query = (
        "Acumatica acumatica.com current owner completed acquisition "
        "majority private equity"
    )
    search.assert_awaited_once()
    fetch.assert_not_awaited()
    assert search.await_args.args[1] == expected_query
    assert result["usage"]["search_calls"] == 1
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert "submitted_source_hints" not in document["prior_observations"]
    assert "server_public_stage_source_fetch" not in document
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
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 4, "search_calls": 2, "fetch_calls": 1,
    }
    assert len(searches) == investigator.MAX_SEARCH_CALLS
    fetch.assert_awaited_once()
    assert requests[1]["tool_choice"] == {
        "type": "function", "function": {"name": "search_web"},
    }


def test_three_pe_prefetches_keep_fresh_fetch_budget(monkeypatch):
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
        finding = {
            "target": "stage", "status": "UNPROVEN",
            "observed_value": None, "reason": "no controlling owner proof",
        }
        name = "submit_findings"
        arguments = {"findings": [finding]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}", "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
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
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 1, "search_calls": 1, "fetch_calls": 0,
    }
    search.assert_awaited_once()
    fetch.assert_not_awaited()
    assert requests[0]["tool_choice"] == "required"
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["investigation_limits"]["remaining_fetch_calls"] == 3


ACADEMY_IR_URL = (
    "https://investors.academy.com/news-releases/news-release-details/"
    "academy-sports-outdoors-grows-retail-footprint-two-new-locations"
)
ACADEMY_IR_QUOTE = (
    'March 10, 2026 /PRNewswire/ -- Academy Sports + Outdoors '
    '("Academy" or the "Company") ( Nasdaq: ASO ), a leading full-line '
    'sporting goods and outdoor recreation retailer, is excited to announce '
    'it will open two new locations in North Canton, Ohio and Muskogee, Oklahoma.'
)


@pytest.mark.parametrize("quote", [
    ACADEMY_IR_QUOTE,
    'Academy Sports + Outdoors ( Nasdaq: ASO ) announced quarterly results.',
    'Academy Sports + Outdoors ("Academy") (Nasdaq: ASO) announced quarterly results.',
    'Academy Sports + Outdoors (“Academy” or the “Company”) ( Nasdaq: ASO ) announced quarterly results.',
])
def test_current_issuer_parenthetical_preserves_whitespace_and_quoted_alias(quote):
    assert _validate_stage(
        "Academy Sports + Outdoors", "Public", ACADEMY_IR_URL, quote,
        first_party_domain="academy.com",
    )["status"] == "VERIFIED"
    assert lead_scorer._public_quote_has_bound_market_locator(
        quote, ("academysportsoutdoors", "academysportsandoutdoors"),
    )


@pytest.mark.parametrize("quote", [
    'Academy Sports + Outdoors received funding from ParentCo ( Nasdaq: PCO ).',
    'Other Retailer ( Nasdaq: OTHR ) announced a partnership with Academy Sports + Outdoors.',
    'Academy Sports + Outdoors bonds trade on Nasdaq.',
    'Academy Sports + Outdoors ( Nasdaq: ASO ) completed its initial public offering in 2020.',
    'Academy Sports + Outdoors ( Nasdaq: ASO ) ceased trading and became private.',
])
def test_whitespace_does_not_admit_unrelated_non_equity_or_superseded_ticker(quote):
    assert _validate_stage(
        "Academy Sports + Outdoors", "Public", ACADEMY_IR_URL, quote,
        first_party_domain="academy.com",
    )["status"] == "UNPROVEN"


def test_issuer_parenthetical_unproven_gets_bounded_rereview(monkeypatch):
    from tests.test_public_stage_unproven_rereview import (
        _finding, _prefetched_request,
    )

    result, requests = _prefetched_request(
        monkeypatch,
        company_name="Academy Sports + Outdoors",
        company_url="https://academy.com/",
        evidence_url=ACADEMY_IR_URL,
        evidence_text=ACADEMY_IR_QUOTE,
        findings=[
            _finding(reason="ticker lacks a separate current listed-share sentence"),
            _finding(status="VERIFIED", value="Public", url=ACADEMY_IR_URL,
                     quote=ACADEMY_IR_QUOTE),
        ],
    )
    assert len(requests) == 2
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 0
    correction = json.loads(requests[1]["messages"][-1]["content"])
    assert "without a separate listed/traded-share sentence" in correction["instruction"]
