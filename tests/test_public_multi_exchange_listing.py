"""Current plural-exchange listings must retain issuer and stage guards."""

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.lead_scorer import _registrable_domain


def _validated_public(company: str, url: str, quote: str) -> str:
    domain = _registrable_domain(url)
    finding = investigator._validated_findings(
        {
            "findings": [{
                "target": "stage",
                "status": "VERIFIED",
                "observed_value": "Public",
                "evidence_url": url,
                "evidence_quote": quote,
                "reason": "The issuer has a current equity listing.",
            }],
        },
        targets=("stage",),
        fetched_pages={url: quote},
        fetched_final_urls={url: url},
        first_party_domains={domain},
        identity_names={company.casefold()},
        identity_anchor={
            "submitted_name": company,
            "submitted_domain": domain,
            "observed_name": company,
            "observed_domain": domain,
        },
    )["stage"]
    return finding["status"]


@pytest.mark.parametrize("company,url,quote", [
    (
        "Standard Chartered",
        "https://www.sc.com/uk/2026/05/26/standard-chartered-granted-finma-licences-to-open-new-swiss-representative-office/",
        "Standard Chartered PLC is listed on the London and Hong Kong stock exchanges.",
    ),
    (
        "HSBC",
        "https://www.hsbc.com/investors/investing-in-hsbc/group-structure",
        "HSBC is listed on the London, Hong Kong, New York and Bermuda stock exchanges.",
    ),
])
def test_first_party_issuer_plural_exchange_listing_is_current_public_proof(
    company, url, quote,
):
    assert _validated_public(company, url, quote) == "VERIFIED"


@pytest.mark.parametrize("quote", [
    "OtherCo is listed on the London and Hong Kong stock exchanges.",
    "Acme is a customer of OtherCo, which is listed on the London and Hong Kong stock exchanges.",
    "Acme is a subsidiary of OtherCo, which is listed on the London and Hong Kong stock exchanges.",
    "Acme plans to be listed on the London and Hong Kong stock exchanges.",
    "Acme is conditionally listed on the London and Hong Kong stock exchanges, subject to approval.",
    "Acme was listed on the London and Hong Kong stock exchanges before it delisted.",
    "Acme is no longer listed on the London and Hong Kong stock exchanges.",
    "Acme bonds are listed on the London and Hong Kong stock exchanges.",
    "Acme is listed on the London and Hong Kong stock exchanges. Acme was acquired by Parent Group in 2025.",
    "Parent Group took Acme private in 2025. Acme is listed on the London and Hong Kong stock exchanges.",
])
def test_plural_exchange_listing_preserves_current_issuer_equity_guards(quote):
    assert _validated_public("Acme", "https://acme.example/investors", quote) == "UNPROVEN"
