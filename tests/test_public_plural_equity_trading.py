"""Plural present-tense trading proof keeps the existing source and event guards."""

import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer


FIXTURES = Path(__file__).parent / "fixtures" / "company_evidence"
URL = "https://investors.marqeta.com/resources/investor-faqs"
QUOTE = (
    "Shares of our Class A Common Stock trade on the Nasdaq Global Select "
    "Market under the trading symbol “MQ”."
)


def _validated(quote, *, url=URL, text=None, company="Marqeta", domain="marqeta.com"):
    finding = {
        "target": "stage", "status": "VERIFIED", "observed_value": "Public",
        "evidence_url": url, "evidence_quote": quote, "reason": "Current exchange proof",
    }
    return investigator._validated_findings(
        {"findings": [finding]}, targets=("stage",),
        fetched_pages={url: text or quote}, fetched_final_urls={url: url},
        first_party_domains={domain}, identity_names={company.casefold()},
        identity_anchor={"submitted_name": company, "submitted_domain": domain,
                         "observed_name": company, "observed_domain": domain},
    )["stage"]


def test_exact_retained_marqeta_source_and_original_submission_pass():
    text = (FIXTURES / "marqeta_investor_faq_visible.txt").read_text()
    submitted = json.loads((FIXTURES / "marqeta_public_finding.json").read_text())
    assert len(text) == 5759
    assert submitted["findings"][0]["evidence_quote"] == QUOTE
    finding = investigator._validated_findings(
        submitted, targets=("stage",), fetched_pages={URL: text},
        fetched_final_urls={URL: URL}, first_party_domains={"marqeta.com"},
        identity_names={"marqeta", "marqetainc"},
        identity_anchor={"submitted_name": "Marqeta", "submitted_domain": "marqeta.com",
                         "observed_name": "Marqeta", "observed_domain": "marqeta.com"},
    )["stage"]
    assert finding["status"] == "VERIFIED"
    assert finding["evidence_quote"] == QUOTE
    assert finding["evidence_url"] == URL


@pytest.mark.parametrize("quote", [
    "Our common shares trade on Nasdaq under trading symbol ACME.",
    "Our common shares trade on the New York Stock Exchange under ticker ACME.",
    "Acme common shares trade on Nasdaq under ticker ACME.",
    "Acme common stock trades on Nasdaq under ticker ACME.",
    "Acme common stock is traded on Nasdaq under ticker ACME.",
])
def test_present_trading_proof_generalizes_to_other_issuers(quote):
    assert _validated(quote, url="https://acme.example/faq", company="Acme",
                      domain="acme.example")["status"] == "VERIFIED"


@pytest.mark.parametrize("quote", [
    "Our common shares do not trade on Nasdaq under ticker MQ.",
    "Our common shares will trade on Nasdaq under ticker MQ.",
    "Our common shares may trade on Nasdaq under ticker MQ.",
    "Our common shares formerly traded on Nasdaq under ticker MQ.",
    "Our common shares previously traded on Nasdaq under ticker MQ.",
    "Our common shares ceased trading on Nasdaq under ticker MQ.",
    "Our common shares trade on Nasdaq if approval is obtained.",
    "If approval is obtained, our common shares trade on Nasdaq.",
    "Our common shares trade on Nasdaq subject to approval.",
    "Our common shares trade on Nasdaq unless approval is denied.",
    "Our bonds trade on Nasdaq under ticker MQ.",
    "Our debt instruments trade on Nasdaq under ticker MQ.",
    "Our notes trade on Nasdaq under ticker MQ.",
    "Our funds trade on Nasdaq under ticker MQ.",
    "Our ETFs trade on Nasdaq under ticker MQ.",
    "Our common shares trade on Nasdaq. Marqeta has been taken private.",
    "Marqeta completed its initial public offering in 2021.",
])
def test_noncurrent_conditional_and_nonequity_quotes_remain_unproven(quote):
    assert _validated(quote)["status"] == "UNPROVEN"


def test_anonymous_third_party_our_shares_quote_cannot_bind_issuer():
    assert _validated(QUOTE, url="https://unrelated.example/faq")["status"] == "UNPROVEN"


def test_first_party_faq_about_other_issuer_cannot_prove_company_public():
    quote = "OtherCo common shares trade on Nasdaq under ticker OTHR."
    assert _validated(quote)["status"] == "UNPROVEN"


def test_quote_missing_from_actual_source_stays_unproven():
    assert _validated(QUOTE, text="Marqeta operates a payments platform.")["status"] == "UNPROVEN"


@pytest.mark.parametrize("quote", [
    "Marqeta common shares trade on Nasdaq under ticker MQ. Marqeta completed its IPO in 2021.",
    "Marqeta completed its IPO in 2021. Marqeta common shares trade on Nasdaq under ticker MQ.",
])
def test_distinct_current_clause_survives_historical_ipo_elsewhere(quote):
    assert lead_scorer._public_quote_has_bound_market_locator(quote, ["marqeta"])
    assert _validated(quote)["status"] == "VERIFIED"

