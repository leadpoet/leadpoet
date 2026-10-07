"""Recover commercial-source discovery without accepting locator claims."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import company_verification as verification


def _locators(html, domain="acme.example"):
    return verification._homepage_navigation_locators(
        html, final_url=f"https://{domain}/", verified_domain=domain,
    )


def _subscription_source(locators, *, attribute="Sells a subscription security platform",
                         targets=("industry", "required_attribute")):
    return investigator._priority_subscription_navigation_source(
        targets=targets, requested_product_service="A security platform",
        requested_attribute=attribute, homepage_navigation_locators=locators,
    )


def test_captured_mind_menu_retains_terms_with_existing_bounds():
    captured = Path(__file__).parent / "fixtures/company_evidence/mind_homepage_navigation.html"
    locators = _locators(captured.read_text(), "mind.io")
    assert {"url": "https://mind.io/terms-of-use", "label": "Terms of use"} in locators
    assert len(locators) == verification.MAX_HOMEPAGE_NAVIGATION_LOCATORS == 40
    assert sum(len(row["url"]) + len(row["label"]) for row in locators) <= (
        verification.MAX_HOMEPAGE_NAVIGATION_TOTAL_CHARACTERS
    )
    # The actual target list admits the fallback; geography prefetch still wins.
    targets = ("stage", "geography", "industry", "required_attribute")
    assert _subscription_source(locators, targets=targets) == "https://mind.io/terms-of-use"
    assert investigator._priority_headquarters_navigation_source(
        targets, locators,
    ) == "https://mind.io/contact"


def test_pricing_and_company_sources_keep_priority_over_terms():
    html = ('<a href="/terms-of-service">Terms of service</a>'
            '<a href="/company">About us</a><a href="/pricing">Pricing</a>'
            '<a href="/legal/subscription-agreement">Subscription agreement</a>')
    locators = _locators(html)
    assert [row["url"] for row in locators][:2] == [
        "https://acme.example/pricing", "https://acme.example/company",
    ]
    # Pricing remains preferred even if an admitted legal link precedes it.
    assert _subscription_source(list(reversed(locators))) == "https://acme.example/pricing"


@pytest.mark.parametrize("path,label", [
    ("/terms-of-use", "Terms of use"),
    ("/legal/terms-of-service", "Terms of service"),
    ("/legal/subscription-agreement", "Subscription agreement"),
    ("/legal/eula.pdf", "EULA"),
    ("/legal/license-agreement", "License agreement"),
    ("/legal/42", "Software license agreement"),
])
def test_visible_commercial_terms_locator_is_a_subscription_fallback(path, label):
    locators = _locators(f'<a href="{path}">{label}</a>')
    assert _subscription_source(locators) == f"https://acme.example{path}"
    assert _subscription_source(locators, targets=("required_attribute",)) == (
        f"https://acme.example{path}"
    )
    assert _subscription_source(locators, attribute="Sells a security platform") == ""
    assert _subscription_source(locators, targets=("stage",)) == ""


def test_hidden_foreign_privacy_and_editorial_terms_do_not_become_fallbacks():
    html = '''
      <a href="https://foreign.example/terms-of-service">Terms of service</a>
      <div hidden><a href="/terms-of-use">Terms of use</a></div>
      <a aria-hidden="true" href="/eula">EULA</a>
      <a style="display:none" href="/license">License</a>
      <a href="/privacy-policy">Privacy policy</a>
      <a href="/blog/terms-of-service">Subscription terms of service</a>
      <a href="/insights/long-subscription-terms-guide">Subscription terms guide</a>
      <a href="/customer-story">Terms of service</a>
    '''
    locators = _locators(html)
    assert all("foreign.example" not in row["url"] for row in locators)
    assert all(row["url"] not in {
        "https://acme.example/terms-of-use", "https://acme.example/eula",
        "https://acme.example/license",
    } for row in locators)
    assert _subscription_source(locators) == ""


def _finding(target, **changes):
    result = {
        "target": target, "status": "VERIFIED", "observed_value": None,
        "observed_country": "", "observed_state": "", "observed_industry": "",
        "observed_subindustry": "", "activity_role": "supplier_operator",
        "evidence_url": "", "evidence_quote": "", "old_name": "", "new_name": "",
        "old_domain": "", "new_domain": "", "shared_linkedin_slug": "",
        "reason": "Fetched company source supports this fact.",
    }
    result.update(changes)
    return result


@pytest.mark.parametrize("commercial", [True, False], ids=["paid-subscription", "customer-subscriptions-only"])
def test_terms_prefetch_keeps_semantic_authority_and_fetch_budget(monkeypatch, commercial):
    terms_url = "https://acme.example/terms-of-service"
    product_url = "https://acme.example/product"
    product_quote = "Acme supplies a security platform that protects company data."
    terms_quote = (
        "Acme supplies its security platform for annual subscription fees."
        if commercial else
        "Acme protects customer data, including records of their subscriptions."
    )
    calls, requests = [], []

    async def fetch(_session, url, **_kwargs):
        calls.append(url)
        assert url == terms_url
        return {"ok": True, "url": url, "final_url": url, "text": terms_quote}

    async def post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if not commercial and len(requests) == 2:
            return 200, {"choices": [{"message": {"tool_calls": [{
                "id": "search", "type": "function", "function": {
                    "name": "search_web", "arguments": json.dumps({
                        "query": "Acme security platform subscription fees",
                    }),
                },
            }]}}]}
        attribute = _finding(
            "required_attribute", evidence_url=terms_url, evidence_quote=terms_quote,
        ) if commercial else _finding(
            "required_attribute", status="UNPROVEN", activity_role="unresolved",
            reason="The terms discuss customer subscriptions, not Acme recurring fees.",
        )
        arguments = {"findings": [
            _finding("industry", observed_industry="Security",
                     observed_subindustry="Data security", evidence_url=product_url,
                     evidence_quote=product_quote),
            attribute,
        ]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function", "function": {
                "name": "submit_findings", "arguments": json.dumps(arguments),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry", "required_attribute"), requested_industry="Security",
        requested_product_service="A security platform",
        requested_attribute="Sells a subscription security platform",
        positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [product_url]},
        verified_homepage_identity={"normalized_name": "Acme",
                                    "registrable_dns_domain": "acme.example",
                                    "linkedin_company_slug": "acme"},
        homepage_navigation_locators=_locators(
            '<a href="/terms-of-service">Terms of service</a>',
        ),
        prefetched_pages={product_url: {"final_url": product_url, "text": product_quote}},
    ))
    assert calls == [terms_url]
    assert result["usage"]["fetch_calls"] == 1
    assert result["usage"]["fetch_calls"] <= investigator.MAX_FETCH_CALLS == 3
    assert result["usage"]["search_calls"] <= investigator.MAX_SEARCH_CALLS == 2
    assert result["claims"]["required_attribute"]["status"] == (
        "VERIFIED" if commercial else "UNPROVEN"
    )
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["server_priority_submitted_source"]["url"] == terms_url


def test_terms_locator_cannot_supply_an_unfetched_commercial_quote():
    terms_url = "https://acme.example/terms-of-service"
    result = investigator._validated_findings(
        {"findings": [_finding("required_attribute", evidence_url=terms_url,
                               evidence_quote="Acme sells annual security subscriptions.")]},
        targets=("required_attribute",), fetched_pages={},
        first_party_domains={"acme.example"}, identity_names={"acme"},
    )
    assert result["required_attribute"]["status"] == "UNPROVEN"
    assert result["required_attribute"]["reason"] == "submitted quote was not present in fetched source"
