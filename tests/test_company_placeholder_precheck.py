"""Company names require evidence; a placeholder prefix proves no mismatch."""

import asyncio
import json
from pathlib import Path

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer, pre_checks
from qualification.scoring.company_fit_decision import company_fit_mismatch
from qualification.scoring.competition import _normalized_company, _normalized_icp


def saved_case():
    path = Path(__file__).parent / "fixtures" / "tyche_20261028_aaa_company.json"
    case = json.loads(path.read_text())
    # Only the public company identity is retained from the saved case.
    # All other inputs are synthetic; no frozen ICP or submitted intent is used.
    company = {
        **case["company"],
        "industry": "Transportation",
        "intent_details": "Synthetic activity for an offline company-name test.",
        "intent_signals": [{
            "description": "Synthetic activity for an offline company-name test.",
            "url": "https://fixture.example/announcement",
            "date": "2026-01-01",
            "matched_icp_signal": 0,
        }],
    }
    icp = {
        "icp_id": "synthetic-company-name-test",
        "industry": "Transportation",
        "employee_count": ["5,001-10,000"],
        "country": "United States",
        "geography": "United States",
        "product_service": "Synthetic test service",
        "intent_signals": ["Synthetic activity for an offline company-name test."],
    }
    return (
        CompanyOutput.model_validate(_normalized_company(company)),
        ICPPrompt.model_validate(_normalized_icp(icp)),
    )


def test_saved_tyche_aaa_cooper_name_reaches_company_prechecks():
    company, _ = saved_case()
    assert company.company_name == "AAA Cooper Transportation"
    assert company.company_website == "http://www.aaacooper.com/"
    assert pre_checks._check_company_data_quality(company) == (True, None)
    result = asyncio.run(pre_checks.run_company_zero_checks(company, 0.0, set()))
    assert result.decision == "match"


@pytest.mark.parametrize("placeholder", pre_checks.PLACEHOLDER_PATTERNS)
def test_exact_placeholder_names_still_fail_before_identity(placeholder):
    company, _ = saved_case()
    company = company.model_copy(update={"company_name": " " + placeholder.upper() + " "})
    passed, reason = pre_checks._check_company_data_quality(company)
    assert passed is False
    assert "placeholder text" in reason


@pytest.mark.parametrize("field,value", [
    ("company_name", ""),
    ("company_name", "AAA <Company>"),
    ("company_website", ""),
    ("company_website", "https://example.com"),
    ("company_website", "https://example.org"),
])
def test_other_company_quality_rejections_stay_in_force(field, value):
    company, _ = saved_case()
    company = company.model_copy(update={field: value})
    assert pre_checks._check_company_data_quality(company)[0] is False


def test_real_name_prefix_still_obeys_runtime_and_duplicate_guards():
    company, _ = saved_case()
    duplicate = asyncio.run(pre_checks.run_company_zero_checks(
        company, 0.0, {company.company_name.lower()},
    ))
    late = asyncio.run(pre_checks.run_company_zero_checks(
        company, pre_checks.CONFIG.RUNNING_MODEL_TIMEOUT_SECONDS + 1, set(),
    ))
    assert duplicate.decision == late.decision == "mismatch"
    assert "Duplicate company" in duplicate.reason
    assert "HARD time limit" in late.reason


def test_saved_name_prefix_does_not_bypass_independent_identity(monkeypatch):
    company, icp = saved_case()
    observed = []

    async def mismatched_homepage(name, website, **kwargs):
        observed.append((name, website))
        return company_fit_mismatch("Fetched homepage identifies another company")

    async def must_not_reverify(*args, **kwargs):
        pytest.fail("A proven homepage mismatch must remain terminal")

    # Keep the real precheck and aggregation path; replace only provider work.
    monkeypatch.setattr(lead_scorer, "verify_company_exists", mismatched_homepage)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", must_not_reverify)
    result = asyncio.run(lead_scorer._verify_company_fit(
        company, icp, 0.0, 0.0, set(), require_https_transport=True,
    ))
    assert observed == [(company.company_name, company.company_website)]
    assert result.decision == "mismatch"
    assert result.details["company_fit_dimensions"]["identity"] == "mismatch"
