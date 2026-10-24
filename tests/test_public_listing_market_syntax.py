"""Exact issuer listing formats from October 24 retained scoring evidence."""

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer


MARTEN_URL = (
    "https://www.sec.gov/Archives/edgar/data/799167/"
    "000143774926002192/ex_910928.htm"
)
MARTEN_QUOTE = (
    "Marten Transport, Ltd. (Nasdaq/GS:MRTN) today reported net income of "
    "$3.7 million, or 5 cents per diluted share, for the fourth quarter "
    "ended December 31, 2025"
)
HEARTLAND_URL = (
    "https://www.sec.gov/Archives/edgar/data/799233/"
    "000079923326000039/htld-20260630.htm"
)
HEARTLAND_QUOTE = (
    "HEARTLAND EXPRESS INC. (Exact Name of Registrant as Specified in Its "
    "Charter) Nevada 93-0926999 (State or Other Jurisdiction (I.R.S. Employer "
    "of Incorporation or organization) Identification No.) 901 Heartland Way, "
    "North Liberty, Iowa 52317 (Address of Principal Executive Offices) "
    "(Zip Code) 319 - 645-7060 (Registrant’s telephone number, including area "
    "code ) Securities registered pursuant to Section 12(b) of the Act: "
    "Title of each class Trading Symbol(s) Name of each exchange on which "
    "registered Common Stock, $0.01 par value HTLD NASDAQ"
)


def validate(name, domain, url, quote, *, page=None):
    return investigator._validated_findings(
        {"findings": [{
            "target": "stage", "status": "VERIFIED", "observed_value": "Public",
            "evidence_url": url, "evidence_quote": quote,
        }]},
        targets=("stage",), fetched_pages={url: quote if page is None else page},
        fetched_final_urls={url: url}, first_party_domains={domain},
        identity_names={"".join(c for c in name.casefold() if c.isalnum())},
        identity_anchor={"submitted_name": name, "submitted_domain": domain},
        requested_stage="Public",
    )["stage"]


def decision(finding):
    return lead_scorer._decision_from_observed_stage(
        {"observed_company_stage": "Public", "stage_matches": True,
         "stage_evidence_url": finding["evidence_url"],
         "stage_evidence_quote": finding["evidence_quote"]},
        "public", validated_stage_finding=finding,
    )


@pytest.mark.parametrize("name,domain,url,quote", [
    ("Marten Transport", "marten.com", MARTEN_URL, MARTEN_QUOTE),
    ("Heartland Express", "heartlandexpress.com", HEARTLAND_URL, HEARTLAND_QUOTE),
])
def test_exact_retained_issuer_quote_reaches_normal_stage_match(name, domain, url, quote):
    finding = validate(name, domain, url, quote)
    assert finding["status"] == "VERIFIED", finding["reason"]
    assert decision(finding) == lead_scorer.COMPANY_FIT_MATCH


@pytest.mark.parametrize("tier", ["GS", "GM", "CM"])
def test_known_nasdaq_market_tier_preserves_issuer_binding(tier):
    quote = f"Acme (Nasdaq/{tier}: ACM) announced current quarterly results."
    finding = validate("Acme", "acme.example", "https://acme.example/investors", quote)
    assert finding["status"] == "VERIFIED", finding["reason"]
    assert decision(finding) == lead_scorer.COMPANY_FIT_MATCH
    assert lead_scorer._stage_evidence_supports_observation(
        "public", quote, evidence_url="https://acme.example/investors",
        evidence_final_url="https://acme.example/investors",
        first_party_domains=("acme.example",), identity_names=("Acme",),
    )


@pytest.mark.parametrize("quote", [
    "Acme received funding from OtherCo (Nasdaq/GS: OTHR).",
    "OtherCo (Nasdaq/GS: OTHR) announced an investment in Acme.",
    "Acme plans to list on Nasdaq/GS: ACM.",
    "Acme (Nasdaq/GS: ACM) was delisted yesterday.",
    "Acme (Nasdaq/GS: ACM) was taken private yesterday.",
    "Acme (Nasdaq/GS: ACM) completed its initial public offering in 2018.",
    "Acme (Nasdaq/PRIVATE: ACM) announced current quarterly results.",
    "Acme (NYSE/GS: ACM) announced current quarterly results.",
    "Acme (Nasdaq/GS: 123) announced current quarterly results.",
    "Marten’s common stock is traded on the Nasdaq Global Select Market under the symbol MRTN.",
])
def test_market_tier_does_not_supply_unbound_future_historical_or_negated_listing(quote):
    name = "Marten Transport" if quote.startswith("Marten") else "Acme"
    finding = validate(name, "acme.example", MARTEN_URL, quote)
    assert finding["status"] == "UNPROVEN"
    assert decision(finding) == lead_scorer.COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize("quote", [
    HEARTLAND_QUOTE.replace("HEARTLAND EXPRESS INC.", "OTHER COMPANY INC."),
    HEARTLAND_QUOTE.replace("Common Stock", "Convertible Notes"),
    HEARTLAND_QUOTE.replace("Section 12(b)", "Section 12(g)"),
    HEARTLAND_QUOTE.replace("HTLD NASDAQ", "HTLD Private Market"),
    HEARTLAND_QUOTE.replace("HTLD NASDAQ", "HTLD NASDAQ Private Market"),
    HEARTLAND_QUOTE.replace("HTLD NASDAQ", "HTLD NASDAQ Other Market"),
    HEARTLAND_QUOTE + ". Heartland Express was taken private.",
    HEARTLAND_QUOTE + ". Heartland Express is no longer listed.",
])
def test_plain_nasdaq_sec_table_keeps_entity_equity_and_chronology_guards(quote):
    finding = validate("Heartland Express", "heartlandexpress.com", HEARTLAND_URL, quote)
    assert finding["status"] == "UNPROVEN"
    assert decision(finding) == lead_scorer.COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize("url", [
    "https://sec.gov.example/Archives/edgar/data/799233/htld.htm",
    "https://www.sec.gov/news/htld.htm", "https://example.com/filing",
])
def test_plain_nasdaq_cover_table_keeps_sec_source_authority(url):
    finding = validate("Heartland Express", "heartlandexpress.com", url, HEARTLAND_QUOTE)
    assert finding["status"] == "UNPROVEN"


def test_listing_syntax_still_requires_exact_fetched_quote():
    for name, domain, url, quote in (
        ("Marten Transport", "marten.com", MARTEN_URL, MARTEN_QUOTE),
        ("Heartland Express", "heartlandexpress.com", HEARTLAND_URL, HEARTLAND_QUOTE),
    ):
        finding = validate(name, domain, url, quote, page="No registered equity evidence.")
        assert finding["status"] == "UNPROVEN"


def _company(name, domain):
    from gateway.qualification.models import CompanyOutput

    return CompanyOutput(
        company_name=name, company_website=f"https://{domain}",
        company_linkedin="", industry="Transportation", employee_count="51-200",
        country="United States", intent_signals=[{
            "description": "hiring", "source": "company website",
            "url": f"https://{domain}/careers", "date": "2026-09-01",
            "snippet": f"{name} is hiring.",
        }],
    )


def test_marten_exact_market_tier_also_reaches_initial_lead_scorer_path():
    verdict = {
        "observed_company_name": "Marten Transport",
        "observed_company_website": "https://marten.com",
        "observed_company_stage": "Public", "stage_matches": True,
        "stage_evidence_url": MARTEN_URL, "stage_evidence_quote": MARTEN_QUOTE,
    }
    assert lead_scorer._decision_from_observed_stage(
        verdict, "public", company=_company("Marten Transport", "marten.com"),
    ) == lead_scorer.COMPANY_FIT_MATCH


def test_sec_table_still_requires_the_server_validated_listing_finding():
    assert lead_scorer._decision_from_observed_stage(
        {"observed_company_stage": "Public", "stage_matches": True,
         "stage_evidence_url": HEARTLAND_URL,
         "stage_evidence_quote": HEARTLAND_QUOTE},
        "public", company=_company("Heartland Express", "heartlandexpress.com"),
    ) == lead_scorer.COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize("name,domain,url,quote", [
    ("Marten Transport", "marten.com", MARTEN_URL, MARTEN_QUOTE),
    ("Heartland Express", "heartlandexpress.com", HEARTLAND_URL, HEARTLAND_QUOTE),
])
def test_validated_retained_listing_cannot_override_current_completed_take_private(
    name, domain, url, quote,
):
    finding = validate(name, domain, url, quote)
    assert finding["status"] == "VERIFIED"
    verdict = {
        "observed_company_name": name, "observed_company_stage": "Public",
        "stage_matches": True, "stage_evidence_url": url,
        "stage_evidence_quote": quote,
        "required_attribute_evidence_url": f"https://{domain}/investors",
        "required_attribute_evidence_quote": f"{name} was taken private yesterday.",
    }
    assert lead_scorer._decision_from_observed_stage(
        verdict, "public", company=_company(name, domain),
        validated_stage_finding=finding,
    ) == lead_scorer.COMPANY_FIT_UNAVAILABLE


def test_plain_nasdaq_cover_row_can_precede_the_next_sec_form_heading():
    quote = HEARTLAND_QUOTE + " Indicate by check mark whether the Registrant has filed all reports."
    finding = validate("Heartland Express", "heartlandexpress.com", HEARTLAND_URL, quote)
    assert finding["status"] == "VERIFIED"
    assert decision(finding) == lead_scorer.COMPANY_FIT_MATCH
