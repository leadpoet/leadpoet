"""Registered-equity evidence uses the same exact-source stage boundary."""

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer


URL = (
    "https://www.sec.gov/Archives/edgar/data/1777319/"
    "000149315226038417/form10-q.htm"
)
# CISO's August 2026 10-Q cover page. Identity and row are one continuous span.
QUOTE = (
    "CISO GLOBAL, INC. (Exact name of registrant as specified in its charter) "
    "Delaware 83-4210278 (State or other Jurisdiction of Incorporation or "
    "Organization) (I.R.S. Employer Identification No.) "
    "6900 E. Camelback Road, Suite 900, Scottsdale, Arizona 85251 "
    "(Address of Principal Executive Offices) (Zip Code) (480) 389-3444 "
    "(Registrant’s telephone number, including area code) "
    "Securities registered pursuant to Section 12(b) of the Act: "
    "Title of each class Trading Symbol(s) Name of each exchange on which "
    "registered Common Stock, $0.00001 par value CISO The Nasdaq Stock Market LLC"
)


def validate(quote=QUOTE, url=URL, names=None, page=None):
    return investigator._validated_findings(
        {"findings": [{
            "target": "stage", "status": "VERIFIED", "observed_value": "Public",
            "evidence_url": url, "evidence_quote": quote,
        }]},
        targets=("stage",), fetched_pages={url: page if page is not None else quote},
        first_party_domains={"ciso.inc"},
        identity_names=names if names is not None else {"cisoglobal"},
    )["stage"]


def test_exact_ciso_sec_table_projects_through_normal_stage_decision():
    finding = validate()
    assert finding["status"] == "VERIFIED"
    verdict = {
        "observed_company_stage": "Public", "stage_matches": True,
        "stage_evidence_url": URL, "stage_evidence_quote": QUOTE,
    }
    assert lead_scorer._decision_from_observed_stage(
        verdict, "public", validated_stage_finding=finding,
    ) == lead_scorer.COMPANY_FIT_MATCH
    # A model-authored quote alone does not produce the server finding.
    assert lead_scorer._decision_from_observed_stage(
        verdict, "public",
    ) == lead_scorer.COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize("quote", [
    QUOTE.replace("CISO The Nasdaq Stock Market LLC", "CISO New York Stock Exchange"),
    QUOTE.replace("Common Stock", "Class A common stock"),
    QUOTE.replace("Common Stock", "Ordinary shares"),
    QUOTE.replace("The Nasdaq Stock Market LLC", "Nasdaq Global Select Market"),
])
def test_registered_equity_table_variants(quote):
    assert validate(quote)["status"] == "VERIFIED"


@pytest.mark.parametrize("quote", [
    QUOTE.replace("Common Stock", "Convertible Notes"),
    QUOTE.replace("Common Stock", "Preferred Stock"),
    QUOTE.replace("CISO The Nasdaq Stock Market LLC", "None None"),
    QUOTE.replace("Trading Symbol(s)", "Market reference"),
    QUOTE.replace("Section 12(b)", "Section 12(g)"),
    QUOTE.replace("Securities registered", "Securities proposed to be registered"),
    QUOTE + ". CISO Global common stock has ceased trading.",
    QUOTE + ". CISO Global is no longer listed.",
    QUOTE + ". CISO Global was taken private.",
    QUOTE.replace("CISO GLOBAL", "OTHER COMPANY"),
    QUOTE.replace("CISO GLOBAL, INC.", "CISO"),
])
def test_non_equity_absent_planned_superseded_or_wrong_entity_stays_unproven(quote):
    assert validate(quote)["status"] == "UNPROVEN"


@pytest.mark.parametrize("url", [
    "https://sec.gov.example/Archives/edgar/data/1777319/form10-q.htm",
    "https://example.com/ciso",
    "https://www.sec.gov/news/example",
])
def test_sec_table_shape_cannot_be_inherited_from_untrusted_host(url):
    assert validate(url=url)["status"] == "UNPROVEN"


def test_quote_must_exist_in_fetched_page_and_name_exact_investigated_company():
    assert validate(page="No listing evidence.")["status"] == "UNPROVEN"
    assert validate(names={"othercompany"})["status"] == "UNPROVEN"
    assert validate(QUOTE.replace("INC.", "INC. ..."), page=QUOTE)["status"] == "UNPROVEN"


def test_subsidiary_cannot_inherit_its_parent_registrants_listing():
    quote = QUOTE.replace("CISO GLOBAL, INC.", "PARENT HOLDINGS, INC.")
    quote += ". CISO Global is a subsidiary of Parent Holdings."
    assert validate(quote)["status"] == "UNPROVEN"


def test_listing_table_does_not_bypass_archived_private_company_guard():
    assert lead_scorer._is_archived_sec_filing_snapshot(URL)
    assert "an old table does not establish current status" in investigator._SYSTEM_PROMPT
    assert "cannot\nby itself override that current private-company evidence" in investigator._SYSTEM_PROMPT
