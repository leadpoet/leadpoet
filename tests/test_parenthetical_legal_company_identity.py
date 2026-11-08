"""November 7 MEC: a supplied initialism does not replace observed legal identity."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    company_quality_receipt_matches_claim,
    evaluate_company_identity,
)

# Exact submitted and observed identity fields retained in the November 7
# sub-535e61773d9c8ec8064818aba9243554 ICP1 company2 score receipt.
COMPANY = {
    "company_name": "MEC (Mayville Engineering Company, Inc.)",
    "company_website": "http://www.mecinc.com/",
    "company_linkedin": "https://www.linkedin.com/company/mayville-engineering-company/",
}
VERDICT = {
    "observed_company_name": "Mayville Engineering Company, Inc.",
    "observed_company_website": "https://www.mecinc.com/",
    "observed_company_linkedin": "https://www.linkedin.com/company/mayville-engineering-company",
}


def _receipt(company=None, verdict=None, *, quality=True):
    return lead_scorer._web_identity_receipt(
        SimpleNamespace(**(company or COMPANY)), verdict or VERDICT,
        verified_homepage_transport_domain="mecinc.com", company_quality=quality,
    )


@pytest.mark.parametrize("quality", [False, True])
def test_exact_nov7_parenthetical_name_keeps_independently_observed_legal_identity(quality):
    receipt = _receipt(quality=quality)
    assert receipt["decision"] == "match"
    assert receipt["observed_name"] == "mayvilleengineering"
    assert receipt["observed_linkedin_slug"] == "mayville-engineering-company"
    # Persisted quality verification replays the canonical observed name, not
    # the raw spaced legal name. It must reproduce the same strict decision.
    assert company_quality_receipt_matches_claim(receipt, COMPANY)


@pytest.mark.parametrize("field,value", [
    ("observed_company_name", "Mayville Engineering Sales, Inc."),
    ("observed_company_name", "Mayville Engineering Company Asia, Inc."),
    ("observed_company_name", "Mayville Engineering Company Holdings North America"),
    ("observed_company_name", "MEC"),
    ("observed_company_name", "Mayville Engineering Company, Inc. customer services"),
    ("observed_company_website", "https://other.example/"),
    ("observed_company_linkedin", "https://www.linkedin.com/company/mec-inc/"),
    ("observed_company_linkedin", ""),
])
def test_conflicting_entity_domain_or_profile_never_uses_parenthetical_name(field, value):
    verdict = {**VERDICT, field: value}
    assert _receipt(verdict=verdict)["decision"] != "match"


@pytest.mark.parametrize("name", [
    "BAD (Mayville Engineering Company, Inc.)",
    "MEC (Mayville Engineering Company, Inc.) partner",
    "MEC (Mayville Engineering Company, Inc. subsidiary)",
    "MEC (Other Company)",
    "MEC (Mayville (Engineering) Company, Inc.)",
    "MEC (Mayville Engineering Company, Inc.) (Other)",
    "mec (Mayville Engineering Company, Inc.)",
])
def test_arbitrary_or_ambiguous_submitted_alias_never_proves_identity(name):
    assert _receipt(company={**COMPANY, "company_name": name})["decision"] != "match"


def test_reverse_parenthetical_requires_independent_source_and_exact_submitted_profile():
    arguments = {
        "submitted_name": COMPANY["company_name"],
        "submitted_website": COMPANY["company_website"],
        "submitted_linkedin": COMPANY["company_linkedin"],
        "observed_name": VERDICT["observed_company_name"],
        "observed_website": VERDICT["observed_company_website"],
        "observed_linkedin": VERDICT["observed_company_linkedin"],
        "evidence_source": "company_web_reverification", "company_quality": True,
    }
    assert evaluate_company_identity(**arguments)["decision"] == "match"
    assert evaluate_company_identity(**{**arguments, "evidence_source": "submitted_claim"})["decision"] != "match"
    assert evaluate_company_identity(**{**arguments, "submitted_linkedin": ""})["decision"] != "match"
    before = deepcopy(arguments)
    evaluate_company_identity(**arguments)
    assert arguments == before


def test_saved_receipt_cannot_be_reused_for_a_changed_parenthetical_claim():
    receipt = _receipt()
    assert not company_quality_receipt_matches_claim(
        receipt, {**COMPANY, "company_name": "BAD (Mayville Engineering Company, Inc.)"},
    )
