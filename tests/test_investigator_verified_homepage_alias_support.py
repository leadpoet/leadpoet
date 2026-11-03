"""Keep unnamed first-party proof bound to an independently verified alias.

The HQ quote is captured evidence. The explicit homepage alias in the positive
cases is a controlled fixture, not a claim that the historical capture had it.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import company_verification as verification
from qualification.scoring import lead_scorer
from tests.test_company_evidence_investigator import _company, _complete_verdict, _icp


HQ_URL = "https://www.regeneron.com/about/our-company/locations"
HQ_QUOTE = (
    "North America Corporate Headquarters US inquiries 777 Old Saw Mill River "
    "Road Tarrytown, NY 10591 United States"
)
PIPELINE_URL = "https://www.regeneron.com/science/investigational-pipeline"
PIPELINE_QUOTE = (
    "Our pipeline is powered by end-to-end research and development capabilities, "
    "and our portfolio of investigational medicines reflects the true "
    "collaborative spirit of Regeneron."
)


def _homepage():
    # Exact host-projected proof shape used by _verified_homepage_identity_anchor.
    return {
        "normalized_name": "regeneronpharmaceuticals",
        "registrable_dns_domain": "regeneron.com",
        "linkedin_company_slug": "regeneron-pharmaceuticals",
        "verified_homepage_brand_alias": True,
        "verified_brand_name": "Regeneron",
        "verified_legal_name_aliases": ["Regeneron Pharmaceuticals, Inc."],
    }


def _anchor():
    return {
        "submitted_name": "Regeneron",
        "observed_name": "Regeneron Pharmaceuticals, Inc.",
        "verified_name": "regeneronpharmaceuticals",
        "submitted_domain": "regeneron.com",
        "observed_domain": "regeneron.com",
        "verified_domain": "regeneron.com",
        "submitted_linkedin_slug": "regeneron-pharmaceuticals",
        "observed_linkedin_slug": "regeneron-pharmaceuticals",
        "verified_linkedin_slug": "regeneron-pharmaceuticals",
        **{k: v for k, v in _homepage().items() if k.startswith("verified_")},
    }


def test_bound_homepage_alias_can_identify_unnamed_first_party_support():
    assert investigator._required_attribute_support_rejection(
        HQ_URL, HQ_QUOTE,
        fetched_pages={HQ_URL: HQ_QUOTE},
        fetched_final_urls={HQ_URL: HQ_URL},
        first_party_domains={"regeneron.com"},
        identity_names={"regeneron", "regeneronpharmaceuticals"},
        identity_anchor=_anchor(),
    ) is None


@pytest.mark.parametrize("update", [
    {"verified_homepage_brand_alias": False},
    {"verified_homepage_brand_alias": "true"},
    {"verified_brand_name": "Regeneron Consumer"},
    {"verified_legal_name_aliases": []},
    {"verified_legal_name_aliases": ["Other Pharmaceuticals, Inc."]},
    {"verified_legal_name_aliases": ["Regeneron Pharmaceuticals, Inc.", "Other"]},
    {"observed_name": "Regeneron Consumer Products"},
    {"submitted_domain": "other.example"},
    {"observed_domain": "other.example"},
    {"verified_domain": "other.example"},
    {"observed_linkedin_slug": "other"},
    {"verified_linkedin_slug": "other"},
])
def test_homepage_alias_support_rejects_incomplete_or_conflicting_identity(update):
    anchor = {**_anchor(), **update}
    assert investigator._required_attribute_support_rejection(
        HQ_URL, HQ_QUOTE,
        fetched_pages={HQ_URL: HQ_QUOTE},
        fetched_final_urls={HQ_URL: HQ_URL},
        first_party_domains={"regeneron.com"},
        identity_names={"regeneron", "regeneronpharmaceuticals"},
        identity_anchor=anchor,
    ) == investigator.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON


def test_recorded_legacy_alias_without_homepage_proof_stays_unproven():
    anchor = _anchor()
    for key in (
        "verified_homepage_brand_alias", "verified_brand_name",
        "verified_legal_name_aliases",
    ):
        anchor.pop(key)
    anchor["verified_name"] = "regeneron"
    assert investigator._required_attribute_support_rejection(
        HQ_URL, HQ_QUOTE,
        fetched_pages={HQ_URL: HQ_QUOTE},
        fetched_final_urls={HQ_URL: HQ_URL},
        first_party_domains={"regeneron.com"},
        identity_names={"regeneron", "regeneronpharmaceuticals"},
        identity_anchor=anchor,
    ) == investigator.REQUIRED_ATTRIBUTE_SUPPORT_REJECTION_REASON


def _mock_investigation(monkeypatch):
    finding = {
        "target": "required_attribute", "status": "VERIFIED",
        "observed_value": None, "observed_country": "", "observed_state": "",
        "observed_industry": "", "observed_subindustry": "",
        "activity_role": "supplier_operator",
        "evidence_url": PIPELINE_URL, "evidence_quote": PIPELINE_QUOTE,
        "supporting_evidence_url_1": HQ_URL,
        "supporting_evidence_quote_1": HQ_QUOTE,
        "supporting_evidence_url_2": "", "supporting_evidence_quote_2": "",
        "old_name": "", "new_name": "", "old_domain": "", "new_domain": "",
        "shared_linkedin_slug": "", "reason": "Own clinical pipeline and Northeast headquarters.",
    }

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "attribute", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({"findings": [finding]})},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    monkeypatch.setattr(investigator, "_fetch_page", AsyncMock(return_value={"text": ""}))


def test_investigator_carries_server_alias_proof_into_support_validation(monkeypatch):
    _mock_investigation(monkeypatch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "Regeneron", "website": "https://www.regeneron.com/",
            "linkedin": "https://www.linkedin.com/company/regeneron-pharmaceuticals/",
        },
        targets=("required_attribute",),
        requested_attribute="Runs an in-house pipeline with a Northeast U.S. headquarters.",
        verified_homepage_identity=_homepage(),
        prior_observations={
            "submitted_source_urls": [PIPELINE_URL, HQ_URL],
            "observed_company_name": "Regeneron Pharmaceuticals, Inc.",
            "observed_company_website": "https://www.regeneron.com/",
            "observed_company_linkedin": "https://www.linkedin.com/company/regeneron-pharmaceuticals/",
        },
        prefetched_pages={url: {"final_url": url, "text": quote} for url, quote in (
            (PIPELINE_URL, PIPELINE_QUOTE), (HQ_URL, HQ_QUOTE),
        )},
    ))
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["supporting_evidence"] == [
        {"url": HQ_URL, "quote": HQ_QUOTE},
    ]


@pytest.mark.parametrize("quality", [False, True])
@pytest.mark.parametrize("case", [
    "verified", "missing_marker", "wrong_domain", "wrong_slug",
    "unverified_alias", "redirect", "provider_only_proof", "missing_observed_linkedin",
])
def test_exact_homepage_through_current_caller_preserves_alias_proof(
    monkeypatch, quality, case,
):
    legal = "Regeneron Pharmaceuticals, Inc."
    website = "https://www.regeneron.com/"
    linkedin = "https://www.linkedin.com/company/regeneron-pharmaceuticals/"
    html = '<script type="application/ld+json">' + json.dumps({
        "@type": "Organization", "name": "Regeneron", "legalName": legal,
        "url": website, "sameAs": [linkedin],
    }) + '</script>'
    homepage_fetch = AsyncMock(return_value=(200, website, html))
    monkeypatch.setattr(verification, "_fetch_bounded_html", homepage_fetch)
    homepage = asyncio.run(verification.verify_company_exists(
        legal, website, company_linkedin=linkedin, require_https_transport=True,
        company_quality=quality,
    ))
    anchor = lead_scorer._verified_homepage_identity_anchor(homepage)
    assert anchor["verified_homepage_brand_alias"] is True
    if case in {"missing_marker", "provider_only_proof"}:
        anchor.pop("verified_homepage_brand_alias")
    elif case == "wrong_domain":
        anchor["registrable_dns_domain"] = "other.example"
    elif case == "wrong_slug":
        anchor["linkedin_company_slug"] = "regeneron-consumer"
    elif case == "unverified_alias":
        anchor["verified_legal_name_aliases"] = ["Other Pharmaceuticals, Inc."]
    company = _company(name=legal, website=website, linkedin=linkedin).model_copy(update={
        "fit_evidence_urls": [PIPELINE_URL, HQ_URL], "intent_signals": [],
    })
    attribute = "Runs an in-house pipeline with a Northeast U.S. headquarters."
    icp = _icp(required_attribute=attribute)
    verdict = _complete_verdict(
        observed_company_name="Regeneron", observed_company_website=website,
        observed_company_linkedin=linkedin,
    )
    if case == "missing_observed_linkedin":
        verdict["observed_company_linkedin"] = ""
    web_receipt = lead_scorer._web_identity_receipt(
        company, verdict, verified_homepage_identity=anchor, company_quality=quality,
    )
    expected_verified = case == "verified" or (
        case == "missing_observed_linkedin" and not quality
    )
    if expected_verified or case == "redirect":
        assert web_receipt["decision"] == "match"
    prior = lead_scorer.company_fit_unavailable("attribute unavailable", details={
        "identity_decision": web_receipt["decision"], "identity_receipt": web_receipt,
    })
    assert lead_scorer._complete_verified_attribute_recovery_identity(
        company, prior, anchor, HQ_URL,
    ) is (expected_verified or case == "redirect")
    if case == "provider_only_proof":
        verdict.update(_homepage())
    _mock_investigation(monkeypatch)
    captured = {}

    async def actual_investigation(**arguments):
        captured["arguments"] = arguments
        captured["result"] = await investigator.investigate_company_evidence(**arguments)
        return captured["result"]
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", actual_investigation)
    asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=company, icp=icp, verdict=verdict,
        investigation_targets=("required_attribute",), icp_attribute=attribute,
        icp_stage="", verified_identity=anchor, verified_transport_domain="regeneron.com",
        structured_employee_size_evidence=None, structured_public_company_evidence=None,
        employee_size_conflict=False, company_quality=quality,
        required_attribute_source_cache={url: {
            "status": "fetched", "text": quote,
            "final_url": (
                "https://other.example/locations"
                if case == "redirect" and url == HQ_URL else url
            ),
        } for url, quote in ((PIPELINE_URL, PIPELINE_QUOTE), (HQ_URL, HQ_QUOTE))},
    ))
    assert captured["arguments"]["verified_homepage_identity"] == anchor
    claims = captured["result"]["claims"]
    if expected_verified:
        assert claims["required_attribute"]["status"] == "VERIFIED"
        assert claims["required_attribute"]["supporting_evidence"] == [
            {"url": HQ_URL, "quote": HQ_QUOTE},
        ]
    else:
        assert claims.get("required_attribute", {}).get("status") != "VERIFIED"
    homepage_fetch.assert_awaited_once()
