"""A server-proved brand can bind its exact legal claim through host validation."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import company_verification as verification
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    CompanyFitDecisionResult, company_quality_receipt_matches_claim,
)
from qualification.scoring.competition import company_quality_identity_verified


LEGAL = "Northstar Residential Holdings, Inc."
BRAND = "Blue Haven"
URL = "https://acme.com/"
LINKEDIN = "https://www.linkedin.com/company/northstar-holdings"


def _company(**changes):
    return CompanyOutput.model_construct(**{
        "company_name": LEGAL, "company_website": URL,
        "company_linkedin": LINKEDIN, **changes,
    })


def _anchor(**changes):
    return {
        "normalized_name": "northstarresidential",
        "registrable_dns_domain": "acme.com",
        "linkedin_company_slug": "northstar-holdings",
        "verified_legal_name_aliases": [LEGAL],
        "verified_brand_name": BRAND, **changes,
    }


def _web(**changes):
    return {
        "observed_company_name": BRAND, "observed_company_website": URL,
        "observed_company_linkedin": LINKEDIN, **changes,
    }


@pytest.mark.parametrize("quality", [False, True])
@pytest.mark.parametrize("rebrand", [None, {"status": "UNPROVEN"}])
def test_saved_maa_brand_receipt_binds_original_legal_claim(quality, rebrand):
    fixture = json.loads((Path(__file__).parent / "fixtures/verified_homepage_brand_maa.json").read_text())
    before = deepcopy(fixture)
    company = CompanyOutput.model_construct(**fixture["company"])
    homepage = CompanyFitDecisionResult(**fixture["homepage"])
    anchor = lead_scorer._verified_homepage_identity_anchor(homepage)
    receipt = lead_scorer._web_identity_receipt(
        company, fixture["web_observations"], verified_homepage_identity=anchor,
        verified_rebrand_identity=rebrand, company_quality=quality,
    )
    assert receipt["decision"] == "match"
    assert receipt["observed_name"] == "maa"
    assert receipt["verified_brand_name"] == "MAA"
    assert receipt["verified_legal_name_aliases"] == ["Mid-America Apartment Communities, Inc."]
    assert company_quality_receipt_matches_claim(receipt, fixture["company"])
    assert company_quality_identity_verified(receipt)
    assert fixture == before


@pytest.mark.parametrize("quality", [False, True])
def test_generic_homepage_organization_brand_reaches_canonical_host_receipt(monkeypatch, quality):
    record = {"@type": "Organization", "name": BRAND, "legalName": LEGAL,
              "url": URL, "sameAs": [LINKEDIN]}
    html = '<script type="application/ld+json">' + json.dumps(record) + '</script>'
    fetch = AsyncMock(return_value=(200, URL, html))
    monkeypatch.setattr(verification, "_fetch_bounded_html", fetch)
    homepage = asyncio.run(verification.verify_company_exists(
        LEGAL, URL, company_linkedin=LINKEDIN, require_https_transport=True,
        company_quality=quality,
    ))
    assert homepage.decision == "match"
    anchor = lead_scorer._verified_homepage_identity_anchor(homepage)
    assert anchor["verified_brand_name"] == BRAND
    receipt = lead_scorer._web_identity_receipt(
        _company(), _web(), verified_homepage_identity=anchor, company_quality=quality,
    )
    assert receipt["decision"] == "match"
    assert receipt["observed_name"] == "bluehaven"
    assert receipt["submitted_name"] == "northstarresidential"
    assert company_quality_receipt_matches_claim(receipt, _company().model_dump())
    assert company_quality_identity_verified(receipt)
    fetch.assert_awaited_once()


@pytest.mark.parametrize("quality", [False, True])
@pytest.mark.parametrize("anchor_changes", [
    {"verified_brand_name": ""}, {"verified_brand_name": None},
    {"verified_brand_name": "Sibling Brand"},
    {"verified_legal_name_aliases": []}, {"verified_legal_name_aliases": None},
    {"verified_legal_name_aliases": ["Unrelated Parent Holdings Inc."]},
    {"verified_legal_name_aliases": [LEGAL, "Sibling Residential Holdings Inc."]},
    {"verified_legal_name_aliases": [LEGAL] * 4},
    {"normalized_name": "unrelatedparentholdings"},
    {"registrable_dns_domain": "other.example"},
    {"linkedin_company_slug": "unrelated-parent"},
])
def test_missing_contradictory_or_unbound_brand_proof_cannot_accept(quality, anchor_changes):
    receipt = lead_scorer._web_identity_receipt(
        _company(), _web(), verified_homepage_identity=_anchor(**anchor_changes),
        company_quality=quality,
    )
    assert receipt["decision"] != "match"


@pytest.mark.parametrize("quality", [False, True])
@pytest.mark.parametrize("company_changes,web_changes", [
    ({"company_name": "Sibling Residential Holdings Inc."}, {}),
    ({"company_website": "https://other.example/"}, {}),
    ({"company_linkedin": "https://www.linkedin.com/company/sibling"}, {}),
    ({"company_linkedin": ""}, {}),
    ({}, {"observed_company_name": "Sibling Brand"}),
    ({}, {"observed_company_website": "https://other.example/"}),
    ({}, {"observed_company_linkedin": "https://www.linkedin.com/company/sibling"}),
    ({}, {"observed_company_linkedin": ""}),
])
def test_brand_does_not_override_name_domain_or_profile_conflicts(quality, company_changes, web_changes):
    receipt = lead_scorer._web_identity_receipt(
        _company(**company_changes), _web(**web_changes),
        verified_homepage_identity=_anchor(), company_quality=quality,
    )
    assert receipt["decision"] != "match"


def test_model_brand_and_alias_fields_do_not_supply_homepage_proof():
    web = _web(verified_brand_name=BRAND, verified_legal_name_aliases=[LEGAL])
    receipt = lead_scorer._web_identity_receipt(_company(), web, company_quality=True)
    assert receipt["decision"] != "match"
    assert "verified_brand_name" not in receipt
    assert "verified_legal_name_aliases" not in receipt


def _brand_receipt():
    return {
        "decision": "match", "reason_code": "verified_homepage_brand_alias",
        "evidence_source": "company_web_reverification",
        "submitted_name": "northstarresidential", "submitted_domain": "acme.com",
        "submitted_linkedin_slug": "northstar-holdings",
        "observed_name": "bluehaven", "observed_domain": "acme.com",
        "observed_linkedin_slug": "northstar-holdings",
        "verified_brand_name": BRAND, "verified_legal_name_aliases": [LEGAL],
    }


def test_host_validates_canonical_verified_brand_receipt():
    assert company_quality_receipt_matches_claim(_brand_receipt(), _company().model_dump())


@pytest.mark.parametrize("changes", [
    {"verified_brand_name": ""}, {"verified_brand_name": "Sibling Brand"},
    {"verified_legal_name_aliases": []}, {"verified_legal_name_aliases": [BRAND]},
    {"verified_legal_name_aliases": [LEGAL, "Sibling Residential Holdings Inc."]},
    {"verified_legal_name_aliases": [LEGAL] * 4},
    {"reason_code": "verifier_accepted"},
    {"submitted_name": "siblingresidentialholdings"},
    {"observed_name": BRAND}, {"observed_domain": "other.example"},
    {"observed_linkedin_slug": "sibling"},
])
def test_host_rejects_incomplete_conflicting_or_noncanonical_brand_receipts(changes):
    assert not company_quality_receipt_matches_claim(
        {**_brand_receipt(), **changes}, _company().model_dump(),
    )
