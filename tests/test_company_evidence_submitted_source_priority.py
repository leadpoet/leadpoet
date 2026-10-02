from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

from qualification.scoring import company_evidence_investigator as investigator


def _finding(url: str, quote: str) -> dict[str, object]:
    return {
        "target": "industry",
        "status": "VERIFIED",
        "observed_value": "Industrial inspection cameras",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "Hardware",
        "observed_subindustry": "Industrial inspection cameras",
        "activity_role": "supplier_operator",
        "evidence_url": url,
        "evidence_quote": quote,
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "Acme builds and sells physical products for industrial use.",
    }


def _unproven_stage_finding() -> dict[str, object]:
    return {
        "target": "stage",
        "status": "UNPROVEN",
        "observed_value": None,
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": "",
        "evidence_quote": "",
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "Later funding evidence does not prove a priced equity round.",
    }


def _tool_response(
    arguments: dict[str, object],
    *,
    name: str = "submit_findings",
    turn: int = 1,
) -> tuple[int, dict[str, object]]:
    return 200, {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": f"call-{turn}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            },
        }],
    }


def _set_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")


def test_industry_priority_skips_historical_stage_source():
    stage_url = "https://acme.example/news/series-a"
    expansion_url = "https://acme.example/news/manufacturing-expansion"
    product_url = "https://acme.example/products/new-camera"

    selected = investigator._priority_submitted_company_source(
        targets=("stage", "industry"),
        requested_private_equity_stage=False,
        submitted_source_urls=(stage_url, expansion_url),
        submitted_stage_source_urls=(stage_url,),
        first_party_domains={"acme.example"},
    )

    assert selected == expansion_url
    assert investigator._priority_submitted_company_source(
        targets=("stage",),
        requested_private_equity_stage=False,
        submitted_source_urls=(stage_url,),
        submitted_stage_source_urls=(stage_url,),
        first_party_domains={"acme.example"},
    ) == ""
    assert investigator._priority_submitted_company_source(
        targets=("stage",),
        requested_private_equity_stage=True,
        submitted_source_urls=(product_url, stage_url),
        submitted_stage_source_urls=(stage_url,),
        first_party_domains={"acme.example"},
    ) == stage_url


def test_positive_industry_review_prefetches_submitted_first_party_source(
    monkeypatch,
):
    url = "https://acme.example/news/manufacturing-expansion"
    quote = (
        "Acme builds and sells industrial inspection cameras to manufacturers."
    )
    requests: list[dict[str, object]] = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return _tool_response({
            "findings": [_unproven_stage_finding(), _finding(url, quote)],
        })

    fetch = AsyncMock(return_value={
        "ok": True,
        "url": url,
        "final_url": url,
        "text": quote,
    })
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    search = AsyncMock(return_value={
        "results": [{
            "url": "https://acme.example/news/additional-funding",
            "title": "Acme announces additional funding",
        }],
    })
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Acme",
            "website": "https://acme.example/",
        },
        targets=("stage", "industry"),
        requested_stage="Series A",
        requested_industry="Hardware",
        requested_attribute=(
            "Builds and sells physical hardware for industrial use with recent "
            "operational expansion"
        ),
        positive_semantic_review=True,
        prior_observations={
            "submitted_source_urls": [url],
        },
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
    ))

    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["fetch_calls"] == 1
    fetch.assert_awaited_once()
    assert fetch.await_args.args[1] == url
    search.assert_awaited_once()
    document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert document["server_priority_submitted_source"] == {
        "url": url,
        "ok": True,
        "cache_hit": False,
        "notice": "server_selected_first_party_locator_is_untrusted_evidence",
    }
    assert document["prefetched_sources"] == [{"url": url, "text": quote}]
    assert document["investigation_limits"]["remaining_fetch_calls"] == 2
    assert document["server_current_stage_discovery"]["ok"] is True


def test_positive_industry_review_reuses_prefetched_priority_source(monkeypatch):
    url = "https://acme.example/news/manufacturing-expansion"
    quote = (
        "Acme builds and sells industrial inspection cameras to manufacturers."
    )
    requests: list[dict[str, object]] = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return _tool_response({"findings": [_finding(url, quote)]})

    fetch = AsyncMock()
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Acme",
            "website": "https://acme.example/",
        },
        targets=("industry",),
        requested_industry="Hardware",
        positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [url]},
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        prefetched_pages={url: {
            "final_url": url,
            "text": quote,
        }},
    ))

    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["fetch_calls"] == 0
    fetch.assert_not_awaited()
    document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert document["server_priority_submitted_source"]["cache_hit"] is True
    assert document["investigation_limits"]["remaining_fetch_calls"] == 3


def test_subscription_review_prefers_late_admitted_pricing_navigation(monkeypatch):
    submitted_url = "https://acme.example/news/company-update"
    pricing_url = "https://acme.example/pricing"
    quote = "Acme sells its subscription platform to business customers."
    requests: list[dict[str, object]] = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return _tool_response({
            "findings": [{
                **_finding(pricing_url, quote),
                "observed_value": "Subscription platform",
                "observed_industry": "Software",
                "observed_subindustry": "Business subscription software",
            }],
        })

    fetch = AsyncMock(return_value={
        "ok": True,
        "url": pricing_url,
        "final_url": pricing_url,
        "text": quote,
    })
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Acme",
            "website": "https://acme.example/",
        },
        targets=("industry",),
        requested_industry="Software",
        requested_product_service="Subscription software for business teams",
        positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [submitted_url]},
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        homepage_navigation_locators=[
            {
                "url": f"https://acme.example/section-{index}",
                "label": f"Section {index}",
            }
            for index in range(7)
        ] + [{"url": pricing_url, "label": "Pricing"}],
    ))

    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["fetch_calls"] == 1
    fetch.assert_awaited_once()
    assert fetch.await_args.args[1] == pricing_url
    document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert document["server_priority_submitted_source"]["url"] == pricing_url


def test_subscription_navigation_prefetch_requires_explicit_intent_and_valid_locator(
    monkeypatch,
):
    submitted_url = "https://acme.example/news/company-update"
    pricing_url = "https://acme.example/pricing"
    unrelated_url = "https://acme.example/integrations"
    quote = "Acme sells software to business customers."
    requests: list[dict[str, object]] = []
    fetched_urls: list[str] = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return _tool_response({"findings": [_finding(submitted_url, quote)]})

    async def fake_fetch(_session, url, **_kwargs):
        fetched_urls.append(url)
        return {
            "ok": True,
            "url": url,
            "final_url": url,
            "text": quote,
        }

    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    async def run_case(requested_attribute, navigation):
        requests.clear()
        fetched_urls.clear()
        return await investigator.investigate_company_evidence(
            company_locator={
                "name": "Acme",
                "website": "https://acme.example/",
            },
            targets=("industry",),
            requested_industry="Software",
            requested_attribute=requested_attribute,
            positive_semantic_review=True,
            prior_observations={"submitted_source_urls": [submitted_url]},
            verified_homepage_identity={
                "normalized_name": "Acme",
                "registrable_dns_domain": "acme.example",
                "linkedin_company_slug": "acme",
            },
            homepage_navigation_locators=navigation,
        )

    asyncio.run(run_case(
        "Software for a business subscription model",
        [
            {"url": unrelated_url, "label": "Integrations"},
            {"url": "https://outside.example/pricing", "label": "Pricing"},
        ],
    ))
    assert fetched_urls == [submitted_url]
    assert "server_priority_submitted_source" in json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )

    asyncio.run(run_case(
        "Software for business teams",
        [{"url": pricing_url, "label": "Pricing"}],
    ))
    assert fetched_urls == [submitted_url]
    assert json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )["server_priority_submitted_source"]["url"] == submitted_url


def test_priority_source_does_not_expand_three_page_prefetch_bound(monkeypatch):
    priority_url = "https://acme.example/news/manufacturing-expansion"
    prefetched_urls = [
        f"https://evidence{index}.example/acme" for index in range(3)
    ]
    requests: list[dict[str, object]] = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 2:
            return _tool_response(
                {"query": "Acme industrial hardware expansion"},
                name="search_web",
                turn=turn,
            )
        return _tool_response(
            {"findings": [{
                **_finding(priority_url, "unused"),
                "status": "UNPROVEN",
                "observed_value": None,
                "observed_industry": "",
                "observed_subindustry": "",
                "activity_role": "unresolved",
                "evidence_url": "",
                "evidence_quote": "",
                "reason": "No complete first-party proof was loaded.",
            }]},
            turn=turn,
        )

    fetch = AsyncMock()
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(
        investigator,
        "_search_web",
        AsyncMock(return_value={"results": []}),
    )

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Acme",
            "website": "https://acme.example/",
        },
        targets=("industry",),
        requested_industry="Hardware",
        positive_semantic_review=True,
        prior_observations={
            "submitted_source_urls": [*prefetched_urls, priority_url],
        },
        prefetched_pages={
            url: {"final_url": url, "text": f"Evidence page {index}."}
            for index, url in enumerate(prefetched_urls)
        },
    ))

    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["usage"]["prefetched_pages"] == 3
    assert result["usage"]["fetch_calls"] == 0
    fetch.assert_not_awaited()
    document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert len(document["prefetched_sources"]) == 3
    assert "server_priority_submitted_source" not in document
