"""Current exchange quote cards can prove Public without weakening history guards."""

from pathlib import Path

import pytest

from qualification.scoring.company_evidence_investigator import _validated_findings
from qualification.scoring.company_verification import current_exchange_profile_names_issuer
from qualification.scoring.lead_scorer import _stage_evidence_supports_observation


JLL = (
    "NYSE / JLL JONES LANG LASALLE INC 321.91 Stock price decreased by "
    "-2.49 dollars, -0.77 percent -2.49 (-0.77%)"
)
PROLOGIS = (
    "Quote & Chart Chart Prologis Inc. New York Stock Exchange: PLD Events "
    "Price Market Line 10m Volume. Today September 25, 2026 4:03 PM ET "
    "Last 133.05 Volume 2.54m"
)
GLOBAL_PAYMENTS = (
    "NYSE / GPN GLOBAL PAYMENTS INC 82.18 Stock price unchanged by "
    "0.00 dollars, 0.00 percent"
)
GLOBAL_PAYMENTS_URL = "https://www.nyse.com/quote/XNYS:GPN"


@pytest.mark.parametrize(("quote", "url", "domains", "names"), [
    (JLL, "https://www.nyse.com/quote/XNYS:JLL", (), ("JLL",)),
    (PROLOGIS, "https://ir.prologis.com/stock/quote-chart",
     ("prologis.com",), ("Prologis", "Prologis, Inc.")),
    (PROLOGIS, "https://ir.prologis.com/stock/quote-chart",
     ("prologis.com",), ("Prologis, Inc.",)),
    ("Acme Holdings Inc. New York Stock Exchange: ACME Last 10.00",
     "https://www.nyse.com/quote/XNYS:ACME", (), ("Acme Holdings",)),
])
def test_current_exchange_profile_is_bound_public_evidence(
    quote, url, domains, names,
):
    assert _stage_evidence_supports_observation(
        "Public", quote, evidence_url=url,
        first_party_domains=domains, identity_names=names,
    )


@pytest.mark.parametrize(("quote", "names"), [
    ("NYSE / JLL ACME HOLDINGS INC 321.91 Stock price decreased by 2 dollars",
     ("JLL",)),
    ("JLL completed its IPO on July 22, 1997 on the New York Stock Exchange.",
     ("JLL",)),
    ("Prologis Inc. New York Stock Exchange: PLD", ("Prologis",)),
    ("Prologis Inc. New York Stock Exchange: PLD Last market update pending",
     ("Prologis",)),
    ("Prologis Inc. New York Stock Exchange: PLD Last 133.05 ceased trading.",
     ("Prologis",)),
    ("Prologis Inc. New York Stock Exchange: PLD Last 133.05 was delisted.",
     ("Prologis",)),
    ("Prologis Inc. New York Stock Exchange: PLD Last 133.05 became private.",
     ("Prologis",)),
    ("Acme Holdings Inc. New York Stock Exchange: ACME Last 10.00",
     ("Acme",)),
    ("Acmeology Inc. New York Stock Exchange: ACMO Last 10.00", ("Acme",)),
    ("Acme Subsidiary Inc. New York Stock Exchange: SUB Last 10.00", ("Acme",)),
])
def test_exchange_profile_rejects_wrong_issuer_history_static_and_supersession(
    quote, names,
):
    assert not _stage_evidence_supports_observation(
        "Public", quote,
        evidence_url="https://www.nyse.com/quote/XNYS:JLL",
        identity_names=names,
    )


def test_exchange_profile_rejects_non_exchange_third_party_source():
    assert not _stage_evidence_supports_observation(
        "Public", JLL,
        evidence_url="https://example.com/quote/JLL",
        identity_names=("JLL",),
    )


def test_unchanged_quote_is_admitted_from_the_exact_loaded_nyse_page():
    page = (Path(__file__).parent / "fixtures" / "nyse_gpn_oct06.txt").read_text()
    finding = _validated_findings(
        {"findings": [{
            "target": "stage",
            "status": "VERIFIED",
            "observed_value": "Public",
            "evidence_url": GLOBAL_PAYMENTS_URL,
            "evidence_quote": GLOBAL_PAYMENTS,
        }]},
        targets=("stage",),
        fetched_pages={GLOBAL_PAYMENTS_URL: page},
        first_party_domains={"globalpayments.com"},
        identity_names={"globalpayments"},
    )["stage"]

    assert finding["status"] == "VERIFIED"
    assert finding["evidence_quote"] == GLOBAL_PAYMENTS


@pytest.mark.parametrize("quote", [
    GLOBAL_PAYMENTS,
    GLOBAL_PAYMENTS.replace("unchanged", "increased").replace("0.00", "1.00"),
    GLOBAL_PAYMENTS.replace("unchanged", "decreased").replace("0.00", "1.00"),
])
def test_current_price_change_direction_does_not_change_public_status(quote):
    assert _stage_evidence_supports_observation(
        "Public", quote, evidence_url=GLOBAL_PAYMENTS_URL,
        identity_names=("Global Payments", "Global Payments Inc."),
    )


@pytest.mark.parametrize(("quote", "url", "domains"), [
    (GLOBAL_PAYMENTS.replace("GLOBAL PAYMENTS", "ACME HOLDINGS"),
     GLOBAL_PAYMENTS_URL, ()),
    (GLOBAL_PAYMENTS + " Global Payments was delisted in 2025.",
     GLOBAL_PAYMENTS_URL, ()),
    (GLOBAL_PAYMENTS + " Global Payments ceased trading.",
     GLOBAL_PAYMENTS_URL, ()),
    (GLOBAL_PAYMENTS.replace("0.00", "pending"), GLOBAL_PAYMENTS_URL, ()),
    ("NYSE Search results: GPN Global Payments Inc. Stock price unchanged by 0.00 dollars.",
     "https://www.nyse.com/search", ()),
    ("NYSE / NYA NYSE COMPOSITE INDEX 82.18 Stock price unchanged by 0.00 dollars.",
     "https://www.nyse.com/index", ()),
    (GLOBAL_PAYMENTS, "https://example.com/quote/GPN", ()),
])
def test_unchanged_price_does_not_bypass_listing_evidence_guards(quote, url, domains):
    assert not _stage_evidence_supports_observation(
        "Public", quote, evidence_url=url, first_party_domains=domains,
        identity_names=("Global Payments", "Global Payments Inc."),
    )


def test_ticker_mention_in_issuer_announcement_is_not_a_current_quote_card():
    assert not current_exchange_profile_names_issuer(
        "Global Payments (NYSE: GPN) announced a new partnership.",
        ("Global Payments", "Global Payments Inc."),
    )
