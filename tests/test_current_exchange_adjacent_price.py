"""Literal market cards with a dollar price retain exact issuer attribution."""

import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.company_verification import current_exchange_profile_names_issuer


FIXTURES = Path(__file__).parent / "fixtures" / "company_evidence"
URL = "https://investors.fiserv.com/news-releases/news-release-details/fiserv-announces-transfer-stock-exchange-listing-nasdaq"
QUOTE = (
    "Fiserv Announces Transfer of Stock Exchange Listing to Nasdaq - Fiserv, Inc. "
    "Skip to content NASDAQ: FISV $45.85"
)


def _validated(quote, *, text=None, url=URL):
    return investigator._validated_findings(
        {"findings": [{"target": "stage", "status": "VERIFIED", "observed_value": "Public",
                       "evidence_url": url, "evidence_quote": quote, "reason": "Current market card"}]},
        targets=("stage",), fetched_pages={url: text or quote}, fetched_final_urls={url: url},
        first_party_domains={"fiserv.com"}, identity_names={"fiserv", "fiservinc"},
        identity_anchor={"submitted_name": "Fiserv", "submitted_domain": "fiserv.com",
                         "observed_name": "Fiserv", "observed_domain": "fiserv.com"},
    )["stage"]


def test_exact_fiserv_source_and_original_model_finding_validate():
    text = (FIXTURES / "fiserv_exchange_transfer_visible.txt").read_text()
    submitted = json.loads((FIXTURES / "fiserv_public_quote_card_finding.json").read_text())
    assert len(text) == 7430
    assert submitted["findings"][0]["evidence_quote"] == QUOTE
    finding = investigator._validated_findings(
        submitted, targets=("stage",), fetched_pages={URL: text}, fetched_final_urls={URL: URL},
        first_party_domains={"fiserv.com"}, identity_names={"fiserv", "fiservinc"},
        identity_anchor={"submitted_name": "Fiserv", "submitted_domain": "fiserv.com",
                         "observed_name": "Fiserv", "observed_domain": "fiserv.com"},
    )["stage"]
    assert finding["status"] == "VERIFIED"
    assert finding["evidence_quote"] == QUOTE


@pytest.mark.parametrize("quote", [
    "Acme, Inc. NASDAQ: ACM $12.50",
    "News Release - Acme, Inc. Skip to content NASDAQ: ACM $12.50",
    "News Release - Acme, Inc. Skip to main content NASDAQ: ACM $12.50",
    "News Release - Acme, Inc. Skip to main navigation NASDAQ: ACM $12.50",
    "Acme plans a product launch in May - Acme, Inc. Skip to content NASDAQ: ACM $12.50",
    "Acme will expand internationally - Acme, Inc. Skip to content NASDAQ: ACM $12.50",
])
def test_adjacent_price_cards_bind_exact_issuer_without_reading_article_topic(quote):
    assert current_exchange_profile_names_issuer(quote, {"acme"})


@pytest.mark.parametrize("quote", [
    "OtherCorp, Inc. NASDAQ: ACM $12.50",
    "Acme acquisition - OtherCorp, Inc. Skip to content NASDAQ: ACM $12.50",
    "Parent Company, Inc. NASDAQ: ACM $12.50",
    "Acme Holdings, Inc. NASDAQ: ACM $12.50",
    "Acme Group, Inc. NASDAQ: ACM $12.50",
    "Acme Mutual Fund NASDAQ: ACM $12.50",
    "Acme expects NASDAQ: ACM $12.50",
    "Acme plans an IPO on NASDAQ: ACM $12.50",
    "If approved, Acme NASDAQ: ACM $12.50",
    "Hypothetically Acme NASDAQ: ACM $12.50",
    "Acme NASDAQ: ACM $12.50 (proposed IPO price)",
    "Acme NASDAQ: ACM $12.50 offering price",
    "Acme NASDAQ: ACM $12.50 expected price",
    "Acme NASDAQ: ACM $12.50 hypothetical price",
    "Archived IPO release Acme NASDAQ: ACM $12.50",
    "Acme NASDAQ: ACM $12.50. Acme was delisted.",
    "Acme NASDAQ: ACM $12.50. Acme common stock ceased trading.",
    "Acme NASDAQ: ACM $12.50. Acme has been taken private.",
    "Acme, Inc. NASDAQ: ACM",
    "Acme, Inc. $12.50 NASDAQ: ACM",
])
def test_unbound_narrative_hypothetical_and_superseded_rows_do_not_qualify(quote):
    assert not current_exchange_profile_names_issuer(quote, {"acme"})


def test_ticker_collision_retains_explicit_different_issuer():
    assert not current_exchange_profile_names_issuer(
        "News - OtherCorp, Inc. Skip to content NASDAQ: ACME $12.50", {"acme"},
    )


@pytest.mark.parametrize("quote", [
    "Acme NASDAQ: ACM ... Rival NASDAQ: RIV $12",
    "Rival NASDAQ: RIV $12 ... Acme NASDAQ: ACM",
    "NASDAQ: ACME ... Rival NASDAQ: RIV $12",
])
def test_other_issuers_price_does_not_admit_unpriced_target(quote):
    assert not current_exchange_profile_names_issuer(quote, {"acme"})


def test_exact_quote_missing_from_source_is_not_admitted():
    assert _validated(QUOTE, text="Fiserv operates a payments platform.")["status"] == "UNPROVEN"


def test_unrelated_source_domain_does_not_get_first_party_card_exception():
    assert _validated(QUOTE, url="https://unrelated.example/news")["status"] == "UNPROVEN"


def test_existing_current_nyse_article_quote_still_validates():
    quote = "Fiserv, Inc. (NYSE: FI), a leading global provider of payments and financial services technology"
    text = (FIXTURES / "fiserv_exchange_transfer_visible.txt").read_text()
    assert _validated(quote, text=text)["status"] == "VERIFIED"
