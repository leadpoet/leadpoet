"""Contact source priority must preserve the existing three-fetch allowance."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.company_verification import _homepage_navigation_locators


@pytest.mark.parametrize(
    ("locators", "expected"),
    [
        ([{"url": "https://acme.example/demo", "label": "Contact Sales"},
          {"url": "https://acme.example/contact-us", "label": "Get in touch"}],
         "https://acme.example/contact-us"),
        ([{"url": "https://another.example/sales", "label": "Contact Sales"},
          {"url": "https://another.example/headquarters", "label": "Our locations"}],
         "https://another.example/headquarters"),
        ([{"url": "https://another.example/company/42", "label": "Contact us"}],
         "https://another.example/company/42"),
        ([{"url": "https://acme.example/contact", "label": "Contact"}],
         "https://acme.example/contact"),
        ([{"url": "https://acme.example/demo", "label": "Contact Sales"}],
         "https://acme.example/demo"),
        ([{"url": "https://acme.example/pricing", "label": "Pricing"}], ""),
        ([], ""),
    ],
    ids=["contact-path-before-demo", "hq-path-before-sales",
         "opaque-contact-label", "plain-contact", "only-demo-fallback",
         "no-eligible-source", "no-locators"],
)
def test_headquarters_source_prefers_contact_path_with_existing_fallbacks(
    locators, expected,
):
    assert investigator._priority_headquarters_navigation_source(
        ("geography",), locators,
    ) == expected
    assert investigator._priority_headquarters_navigation_source(
        ("required_attribute",), locators,
    ) == ""


def _tool_response(name, arguments, turn):
    return 200, {"choices": [{"message": {"tool_calls": [{
        "id": f"call-{turn}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}}]}


def _finding(target, **changes):
    result = {
        "target": target, "status": "VERIFIED", "observed_value": None,
        "observed_country": "", "observed_state": "",
        "observed_industry": "", "observed_subindustry": "",
        "activity_role": "unresolved", "evidence_url": "", "evidence_quote": "",
        "old_name": "", "new_name": "", "old_domain": "", "new_domain": "",
        "shared_linkedin_slug": "", "reason": "Exact company source supports this fact.",
    }
    result.update(changes)
    return result


def test_kestra_contact_prefetch_leaves_pricing_reachable_with_three_fetches(
    monkeypatch,
):
    # These are the relevant visible links and labels, in document order, from
    # the retained October 6 Kestra homepage scrape (provider entry 2114689).
    homepage = (
        '<a href="/pricing">Pricing</a>'
        '<a href="/demo">Contact Sales</a>'
        '<a href="/about-us">About Us</a>'
        '<a href="/contact-us">Contact us</a>'
    )
    locators = _homepage_navigation_locators(
        homepage, final_url="https://kestra.io/", verified_domain="kestra.io",
    )
    contact = "https://kestra.io/contact-us"
    stage = "https://kestra.io/blogs/kestra-series-a"
    pricing = "https://kestra.io/pricing"
    about = "https://kestra.io/about-us"
    industry_quote = (
        "Kestra empowers engineers to orchestrate any workflow, in any language, "
        "on any infrastructure, delivering unparalleled freedom and flexibility."
    )
    stage_quote = "Kestra announced a $25 million Series A on March 31, 2026."
    pricing_quote = (
        "Kestra Enterprise Edition Kestra Cloud Enterprise Edition Built for "
        "mission-critical workloads Operate, govern, and scale workflows across "
        "your entire organization Contact Sales Annual Subscription"
    )
    source_text = {
        contact: "Kestra Contact us Book a demo.",
        stage: stage_quote,
        pricing: pricing_quote,
        "https://kestra.io/demo": "Kestra Book a Demo Contact Sales.",
    }
    calls = []
    requests = []

    async def fetch(_session, url, **_kwargs):
        calls.append(url)
        return {"ok": True, "url": url, "final_url": url,
                "text": source_text[url]}

    async def post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn <= 3:
            # The production reasoning fetched contact, then the round. Its
            # mistaken demo prefetch exhausted the cap before pricing. With
            # contact prefetched, its same contact fetch reuses loaded text.
            if turn == 3:
                assert len(calls) == 2
            return _tool_response("fetch_page", {"url": (contact, stage, pricing)[turn - 1]}, turn)
        return _tool_response("submit_findings", {"findings": [
            _finding("geography", status="UNPROVEN",
                     reason="The contact page does not state headquarters."),
            _finding("stage", observed_value="Series A", evidence_url=stage,
                     evidence_quote=stage_quote),
            _finding("industry", observed_industry="Software",
                     observed_subindustry="workflow orchestration",
                     activity_role="supplier_operator", evidence_url=about,
                     evidence_quote=industry_quote),
            _finding("required_attribute", activity_role="supplier_operator",
                     evidence_url=pricing, evidence_quote=pricing_quote),
        ]}, turn)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Kestra", "website": "https://kestra.io/",
                         "linkedin": "https://www.linkedin.com/company/kestra"},
        targets=("geography", "stage", "industry", "required_attribute"),
        requested_stage="Series A", requested_industry="Software",
        requested_product_service="A subscription workflow software platform.",
        requested_attribute="Sells a subscription software platform used to manage workflows.",
        positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [about, pricing, "https://kestra.io/"]},
        verified_homepage_identity={"normalized_name": "Kestra",
                                    "registrable_dns_domain": "kestra.io",
                                    "linkedin_company_slug": "kestra"},
        homepage_navigation_locators=locators,
        prefetched_pages={about: {"final_url": about, "text": industry_quote},
                          "https://kestra.io/": {"final_url": "https://kestra.io/",
                                                "text": "Kestra workflow orchestration."}},
    ))
    assert calls == [contact, stage, pricing]
    assert result["usage"]["fetch_calls"] == investigator.MAX_FETCH_CALLS == 3
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["evidence_url"] == pricing
    assert result["claims"]["geography"]["status"] == "UNPROVEN"
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["server_priority_submitted_source"]["url"] == contact
    assert document["investigation_limits"]["remaining_fetch_calls"] == 2
