"""The ICP's required_attribute is scorer-enforced.

Previously the model's attribute validation was stripped before scoring and
never checked — a model could fabricate `passed: true` (or skip the work
entirely) with no scoring consequence. The binary fit gate now requires a
claim that exists, passed, and carries evidence whenever the ICP pins an
attribute.
"""

from gateway.qualification.models import (
    CompanyOutput,
    ICPPrompt,
    RequiredAttributeClaim,
)
from qualification.scoring.lead_scorer import _run_autoresearch_binary_fit_checks


def _icp(attribute=""):
    return ICPPrompt(
        icp_id="t-1", prompt="p", industry="Software", sub_industry="SaaS",
        employee_count="11-50|51-200", company_stage="", geography="United States",
        product_service="x", required_attribute=attribute,
    )


def _company(claim=None):
    return CompanyOutput(
        company_name="Acme", company_website="https://acme.com",
        industry="Software", employee_count="51-200", country="United States",
        intent_signals=[{
            "description": "raised a round", "source": "news",
            "url": "https://news.example.com/a", "date": "2026-07-01",
            "snippet": "Acme announced it raised a funding round this month.",
        }],
        required_attribute=claim,
    )


def _ok_claim(**over):
    base = dict(text="privately held", passed=True,
                evidence_url="https://acme.com/about", evidence_quote="q",
                explanation="e")
    base.update(over)
    return RequiredAttributeClaim(**base)


def test_no_icp_attribute_no_gate():
    ok, reason = _run_autoresearch_binary_fit_checks(_company(None), _icp(""))
    assert ok, reason


def test_backed_claim_passes():
    ok, reason = _run_autoresearch_binary_fit_checks(
        _company(_ok_claim()), _icp("The company is privately held"))
    assert ok, reason


def test_missing_claim_fails():
    ok, reason = _run_autoresearch_binary_fit_checks(
        _company(None), _icp("The company is privately held"))
    assert not ok and "Missing required_attribute" in reason


def test_unpassed_claim_fails():
    ok, reason = _run_autoresearch_binary_fit_checks(
        _company(_ok_claim(passed=False)), _icp("The company is privately held"))
    assert not ok and "did not pass" in reason


def test_evidence_free_claim_fails():
    ok, reason = _run_autoresearch_binary_fit_checks(
        _company(_ok_claim(evidence_url="")), _icp("The company is privately held"))
    assert not ok and "no evidence URL" in reason
