"""Bind structured homepage brands without relaxing source or identity gates."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import company_verification as verification
from qualification.scoring.lead_scorer import _verified_homepage_identity_anchor


URL = "https://acmepay.example/"
LINKEDIN = "https://www.linkedin.com/company/acmepay"
RECORD = {
    "@type": "Organization",
    "name": "AcmePay",
    "legalName": "Separate Legal Entity Ltd",
    "url": URL,
    "sameAs": [LINKEDIN],
}


def _page(record, *, title="Payments for the global stage", linkedin=LINKEDIN):
    return (
        f"<html><head><title>{title}</title>"
        '<script type="application/ld+json">'
        + json.dumps(record)
        + '</script></head><body><a href="'
        + linkedin
        + '">LinkedIn</a>'
        "<p>We operate under a global suite of licenses and regulatory approvals, "
        "meeting strict standards wherever you do business.</p></body></html>"
    )


def _verify(monkeypatch, record, *, title="Payments for the global stage",
            linkedin=LINKEDIN, name="AcmePay", url=URL, final_url=None):
    fetch = AsyncMock(return_value=(
        200, final_url or url, _page(record, title=title, linkedin=linkedin),
    ))
    monkeypatch.setattr(verification, "_fetch_bounded_html", fetch)
    pages = {}
    result = asyncio.run(verification.verify_company_exists(
        name, url, company_linkedin="", company_quality=True,
        homepage_evidence_sink=pages,
    ))
    fetch.assert_awaited_once()
    return result, pages


@pytest.mark.parametrize("shape", ["root", "list", "graph"])
def test_bound_root_organization_supplies_missing_brand(monkeypatch, shape):
    record = {
        "root": RECORD,
        "list": [RECORD],
        "graph": {"@graph": [RECORD]},
    }[shape]
    result, _ = _verify(monkeypatch, record)

    assert result.decision == "match"
    assert _verified_homepage_identity_anchor(result)["normalized_name"] == "acmepay"
    assert result.details["identity"]["observed_name"] != "separatelegalentity"


@pytest.mark.parametrize("changes", [
    {"url": "https://unrelated.example/"},
    {"url": ""},
    {"url": None},
    {"url": "/"},
    {"url": "https://user:password@acmepay.example/"},
    {"url": "https://acmepay.example:bad/"},
    {"url": "https://localhost/"},
    {"name": ""},
    {"name": None},
    {"name": {"name": "AcmePay"}},
    {"name": "x" * 201},
    {"sameAs": []},
    {"sameAs": ["https://www.linkedin.com/company/unrelated"]},
    {"sameAs": [LINKEDIN, "https://www.linkedin.com/company/unrelated"]},
    {"@type": "Person"},
])
def test_unbound_or_malformed_organization_does_not_supply_brand(monkeypatch, changes):
    result, _ = _verify(monkeypatch, {**RECORD, **changes})

    assert result.decision == "unavailable"
    assert _verified_homepage_identity_anchor(result) == {}


@pytest.mark.parametrize("relationship", ["publisher", "parentOrganization", "member", "author"])
def test_nested_other_organization_cannot_supply_brand(monkeypatch, relationship):
    result, _ = _verify(monkeypatch, {"@type": "WebPage", relationship: RECORD})

    assert result.decision == "unavailable"


def test_conflicting_bound_root_names_do_not_supply_brand(monkeypatch):
    result, _ = _verify(monkeypatch, [RECORD, {**RECORD, "name": "Other Brand"}])

    assert result.decision == "unavailable"


def test_conflicting_observed_linkedin_cannot_supply_brand(monkeypatch):
    result, _ = _verify(
        monkeypatch, RECORD, linkedin="https://www.linkedin.com/company/unrelated",
    )

    assert result.decision == "unavailable"


def test_legal_name_alone_does_not_supply_operating_brand(monkeypatch):
    record = {key: value for key, value in RECORD.items() if key != "name"}
    result, _ = _verify(monkeypatch, {**record, "legalName": "AcmePay"})

    assert result.decision == "unavailable"


def test_other_company_redirect_cannot_supply_brand(monkeypatch):
    result, _ = _verify(monkeypatch, RECORD, final_url="https://unrelated.example/")

    assert result.decision == "mismatch"


def test_existing_title_and_unbound_helper_behavior_stays_unchanged(monkeypatch):
    result, _ = _verify(monkeypatch, {**RECORD, "url": ""}, title="AcmePay")

    assert result.decision == "match"
    assert verification._homepage_company_names(_page(RECORD)) == [
        "Payments for the global stage",
    ]


def test_bound_organization_name_survives_existing_candidate_limit():
    metadata = "".join(
        f'<meta property="og:site_name" content="Generic tagline {index}">'
        for index in range(12)
    )
    names = verification._homepage_company_names(
        metadata + _page(RECORD),
        observed_domain="acmepay.example", observed_linkedin=LINKEDIN,
    )

    assert names[0] == "AcmePay"
    assert len(names) == 10


@pytest.mark.parametrize("name,domain,slug,legal_name", [
    ("Sokin", "sokin.com", "sokin", "Plata Capital Ltd"),
    ("AcmePay", "acmepay.example", "acmepay", "Separate Legal Entity Ltd"),
])
def test_homepage_receipt_binds_real_investigator_quote_validation(
    monkeypatch, name, domain, slug, legal_name,
):
    """Use the saved Sokin metadata shape; mock only network/model boundaries."""
    url = f"https://{domain}/"
    linkedin = f"https://www.linkedin.com/company/{slug}"
    record = {
        "@type": "Organization", "name": name, "legalName": legal_name,
        "url": url, "sameAs": [linkedin],
    }
    homepage, pages = _verify(
        monkeypatch, record, name=name, url=url, linkedin=linkedin,
        title="Unified global business banking | Payments for the global stage",
    )
    anchor = _verified_homepage_identity_anchor(homepage)
    quote = (
        "We operate under a global suite of licenses and regulatory approvals, "
        "meeting strict standards wherever you do business."
    )
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit-attribute", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": [{
                    "target": "required_attribute", "status": "VERIFIED",
                    "activity_role": "supplier_operator", "evidence_url": url,
                    "evidence_quote": quote,
                    "reason": "The first-party company states it operates under licenses.",
                }],
            })},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    network = AsyncMock(side_effect=AssertionError("unexpected network call"))
    monkeypatch.setattr(investigator, "_fetch_page", network)
    monkeypatch.setattr(investigator, "_search_web", network)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": name, "website": url, "linkedin": ""},
        targets=("required_attribute",),
        requested_attribute="Operates under regulatory licenses.",
        positive_semantic_review=True,
        prior_observations={
            "observed_company_name": name, "observed_company_website": url,
            "observed_company_linkedin": linkedin,
            "submitted_source_urls": [url],
        },
        verified_homepage_identity=anchor, prefetched_pages=pages,
    ))

    assert homepage.decision == "match"
    assert anchor["normalized_name"] == name.casefold()
    assert anchor["registrable_dns_domain"] == domain
    assert anchor["linkedin_company_slug"] == slug
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["usage"]["fetch_calls"] == result["usage"]["search_calls"] == 0
    assert len(requests) == 1
    assert name.casefold() not in quote.casefold()
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["verified_homepage_identity"] == anchor
    network.assert_not_awaited()

    # The identity receipt does not prove a capability or a regulatory fact.
    # Even this fully bound company must cite actual loaded source words.
    forged = investigator._validated_findings(
        {"findings": [{
            "target": "required_attribute", "status": "VERIFIED",
            "activity_role": "supplier_operator", "evidence_url": url,
            "evidence_quote": "We are licensed on Mars.",
        }]},
        targets=("required_attribute",),
        fetched_pages={url: pages[url]["text"]},
        fetched_final_urls={url: url}, first_party_domains={domain},
        identity_names={name.casefold()},
        identity_anchor={
            "submitted_name": name, "observed_name": name,
            "verified_name": anchor["normalized_name"],
            "submitted_domain": domain, "observed_domain": domain,
            "verified_domain": anchor["registrable_dns_domain"],
            "submitted_linkedin_slug": "", "observed_linkedin_slug": slug,
            "verified_linkedin_slug": anchor["linkedin_company_slug"],
        },
    )
    assert forged["required_attribute"]["status"] == "UNPROVEN"
