"""Legal LLP suffixes must not trigger an unnecessary rebrand investigation."""

from types import SimpleNamespace

import pytest

from leadpoet_verifier.identity.normalization import exact_or_legal_name_match
from qualification.scoring.company_fit_decision import evaluate_company_identity
from qualification.scoring.lead_scorer import _web_identity_receipt


@pytest.mark.parametrize("observed_name", ["BPM LLP", "BPM LLC"])
def test_saved_bpm_identity_receipt_accepts_terminal_legal_suffix(observed_name):
    # Identity fields from the October 11 BPM submission and scorer receipt.
    # The model's unrelated, unproven rebrand finding must not make a legal
    # suffix difference unavailable. Stage evidence remains a separate gate.
    company = SimpleNamespace(
        company_name="BPM", company_website="https://bpm.com/", company_linkedin="",
    )
    receipt = _web_identity_receipt(
        company,
        {
            "observed_company_name": observed_name,
            "observed_company_website": "https://www.bpm.com",
            "observed_company_linkedin": "https://www.linkedin.com/company/bpmllp",
        },
        verified_homepage_identity={
            "normalized_name": "bpm",
            "registrable_dns_domain": "bpm.com",
            "linkedin_company_slug": "bpmllp",
        },
        verified_rebrand_identity={
            "status": "UNPROVEN", "old_name": "BPM LLP", "new_name": "BPM",
            "old_domain": "bpm.com", "new_domain": "bpm.com",
            "shared_linkedin_slug": "bpmllp",
        },
        company_quality=True,
    )

    assert receipt["decision"] == "match"
    assert receipt["reason_code"] == "verifier_accepted"
    assert receipt["submitted_name"] == receipt["observed_name"] == "bpm"


def _identity(**updates):
    fields = {
        "submitted_name": "BPM",
        "submitted_website": "https://bpm.com/",
        "submitted_linkedin": "https://www.linkedin.com/company/bpmllp",
        "observed_name": "BPM LLP",
        "observed_website": "https://www.bpm.com",
        "observed_linkedin": "https://www.linkedin.com/company/bpmllp",
        "evidence_source": "company_web_reverification",
        "company_quality": True,
    }
    fields.update(updates)
    return evaluate_company_identity(**fields)


@pytest.mark.parametrize("observed_name", [
    "BPM LLP", "BPM llp", "BPM LLP.", "BPM, LLP", "BPM (LLP)",
])
def test_llp_suffix_spellings_supported_by_shared_normalizer(observed_name):
    assert exact_or_legal_name_match("BPM", observed_name)
    assert _identity(observed_name=observed_name)["decision"] == "match"


@pytest.mark.parametrize("updates", [
    {"observed_name": "BPM Advisory LLP"},
    {"observed_name": "Other LLP"},
    {"observed_name": "LLP BPM"},
    {"observed_name": "BPM L.L.P."},
    {"observed_website": "https://other.com"},
    {"observed_linkedin": "https://www.linkedin.com/company/other"},
])
def test_llp_suffix_does_not_relax_other_identity_requirements(updates):
    assert _identity(**updates)["decision"] != "match"
