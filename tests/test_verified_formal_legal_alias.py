"""Exact legal-name alias controls for verified same-company identity."""

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring.company_verification import (
    verified_brand_formal_legal_name,
)
from qualification.scoring.lead_scorer import _web_identity_receipt


def _company(name="AES", linkedin=""):
    return CompanyOutput(
        company_name=name,
        company_website="https://www.aes.com/",
        company_linkedin=linkedin,
        industry="Energy",
        employee_count="10,001+",
        company_stage="Public",
        country="United States",
        intent_signals=[{
            "description": "AES Indiana began operations at a new site.",
            "source": "news",
            "url": "https://www.aesindiana.com/press-release/energy-center",
            "date": "2026-02-11",
            "snippet": "The energy center is in service.",
        }],
    )


_HOMEPAGE = {
    "normalized_name": "aes",
    "registrable_dns_domain": "aes.com",
    "linkedin_company_slug": "aes",
}


def _receipt(
    observed_name="The AES Corporation",
    *,
    company=None,
    observed_website="https://www.aes.com/",
    observed_linkedin="https://www.linkedin.com/company/aes",
    homepage=_HOMEPAGE,
):
    return _web_identity_receipt(
        company or _company(),
        {
            "observed_company_name": observed_name,
            "observed_company_website": observed_website,
            "observed_company_linkedin": observed_linkedin,
        },
        verified_homepage_identity=homepage,
        verified_rebrand_identity={"status": "UNPROVEN"},
        company_quality=True,
    )


def test_aes_exact_captured_legal_name_is_not_a_rebrand():
    receipt = _receipt()
    assert receipt["decision"] == "match"
    assert receipt["reason_code"] == "verified_formal_legal_name_alias"
    assert receipt["submitted_domain"] == receipt["observed_domain"] == "aes.com"
    assert receipt["observed_linkedin_slug"] == "aes"


@pytest.mark.parametrize("observed_name", [
    "The AES Holdings Corporation",
    "The AES Parent Corporation",
    "AES Resources Corporation",
    "AES Customer Corporation",
    "The AES Group",
    "The AES",  # A leading article alone is not a formal legal name.
    "American Energy Services Corporation",  # Initialisms need separate proof.
])
def test_aes_does_not_inherit_an_extended_or_unproven_name(observed_name):
    assert _receipt(observed_name)["decision"] != "match"


@pytest.mark.parametrize("changes", [
    {"observed_website": "https://other.example/"},
    {"observed_linkedin": "https://www.linkedin.com/company/aes-resources"},
    {"homepage": {}},
    {"homepage": {**_HOMEPAGE, "normalized_name": "aesresources"}},
])
def test_aes_legal_alias_requires_exact_homepage_and_web_anchors(changes):
    assert _receipt(**changes)["decision"] != "match"


def test_aes_alias_rejects_wrong_submitted_linkedin():
    assert _receipt(company=_company(linkedin="https://www.linkedin.com/company/aes-resources"))["decision"] != "match"


def test_formal_legal_alias_is_reversible_when_the_homepage_verifies_legal_name():
    company = _company(name="The AES Corporation")
    receipt = _receipt(
        "AES",
        company=company,
        homepage={**_HOMEPAGE, "normalized_name": "theaes"},
    )
    assert receipt["decision"] == "match"
    assert receipt["reason_code"] == "verified_formal_legal_name_alias"


def test_formal_name_grammar_requires_exact_core_and_terminal_legal_suffix():
    assert verified_brand_formal_legal_name("AES", "The AES Corporation")
    assert not verified_brand_formal_legal_name("Acme Inc", "The Acme Corporation")
    assert not verified_brand_formal_legal_name("AES", "The AES Holdings Corporation")
    assert not verified_brand_formal_legal_name("AES", "The AES Group")
    assert not verified_brand_formal_legal_name("AES", "AES Corporation Customer")


def test_two_distinct_formal_legal_names_do_not_use_the_brand_bridge():
    company = _company(name="AES Inc")
    receipt = _receipt(
        "The AES LLC",
        company=company,
        homepage=_HOMEPAGE,
    )
    assert receipt["decision"] != "match"
