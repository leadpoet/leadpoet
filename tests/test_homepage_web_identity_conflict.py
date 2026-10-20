"""A first-party entity must survive an unbound web identity claim."""

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_verification, lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
    evaluate_company_identity,
)


HOMEPAGE = """<html><head><title>Quality Systems Integrated Corporation</title>
<meta property="og:site_name" content="Quality Systems Integrated Corporation">
<script type="application/ld+json">{"@type":"Organization",
"name":"Quality Systems Integrated Corporation","url":"https://qsic.com/"}</script>
</head><body><a href="https://www.linkedin.com/company/quality-systems-integrated">LinkedIn</a>
<main>We deliver high-reliability electronics manufacturing.</main></body></html>"""


def _company():
    return CompanyOutput(
        company_name="Qsic", company_website="https://qsic.com/",
        company_linkedin="", industry="Advertising", employee_count="51-200",
        country="United States", intent_signals=[{
            "description": "Qsic announced a product", "source": "news",
            "url": "https://news.example/qsic", "date": "2026-08-12",
            "snippet": "Qsic announced a product",
        }],
    )


def _icp():
    return ICPPrompt(
        icp_id="adtech", prompt="adtech", industry="Advertising",
        sub_industry="Advertising technology platform",
        employee_count="51-200", company_stage="",
        geography="United States", country="United States",
        product_service="Digital advertising platform",
    )


def _web_receipt(*, observed_domain="https://qsic.com/", reason="verifier_accepted"):
    receipt = evaluate_company_identity(
        submitted_name="Qsic", submitted_website="https://qsic.com/",
        submitted_linkedin="", observed_name="QSIC",
        observed_website=observed_domain,
        observed_linkedin="https://www.linkedin.com/company/qsic",
        evidence_source="company_web_reverification",
    )
    if receipt["decision"] == COMPANY_FIT_MATCH:
        receipt["reason_code"] = reason
    return receipt


@pytest.mark.parametrize(
    ("html", "status", "expected_anchor"),
    [
        (HOMEPAGE, 200, True),
        (HOMEPAGE.replace(
            "</main>",
            '<a href="https://www.linkedin.com/company/partner">Partner</a></main>',
        ), 200, False),
        (HOMEPAGE.replace(
            '<meta property="og:site_name" content="Quality Systems Integrated Corporation">',
            "",
        ), 200, False),
        ("", 404, False),
    ],
)
def test_only_unambiguous_first_party_organization_is_retained(
    monkeypatch, html, status, expected_anchor,
):
    async def fetch(_session, _url):
        return status, "https://qsic.com/", html

    monkeypatch.setattr(company_verification, "_fetch_bounded_html", fetch)
    homepage = asyncio.run(company_verification.verify_company_exists(
        "Qsic", "https://qsic.com/", company_linkedin="",
        require_https_transport=True,
    ))
    assert homepage.decision == COMPANY_FIT_UNAVAILABLE
    assert bool(homepage.details.get("observed_homepage_organization_identity")) is expected_anchor
    assert lead_scorer._unresolved_homepage_web_identity_conflict(
        homepage, _web_receipt()
    ) is expected_anchor


@pytest.mark.parametrize(
    ("web_reason", "expected"),
    [
        ("verifier_accepted", COMPANY_FIT_UNAVAILABLE),
        ("verified_rebrand_continuity", COMPANY_FIT_MATCH),
    ],
)
def test_qsic_homepage_conflict_prevents_only_unbound_web_identity(
    monkeypatch, web_reason, expected,
):
    async def fetch(_session, _url):
        return 200, "https://qsic.com/", HOMEPAGE

    async def web(*_args, **_kwargs):
        evidence = {key: {"url": "https://getqsic.com/", "quote": "QSIC"}
                    for key in ("employee_size", "industry", "geography")}
        return company_fit_match("web claims fit", details={
            "identity_decision": COMPANY_FIT_MATCH,
            "identity_receipt": _web_receipt(reason=web_reason),
            "dimension_decisions": {
                "employee_size": COMPANY_FIT_MATCH,
                "industry": COMPANY_FIT_MATCH,
                "geography": COMPANY_FIT_MATCH,
            },
            "dimension_evidence": evidence,
        })

    monkeypatch.setattr(company_verification, "_fetch_bounded_html", fetch)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web)
    result = asyncio.run(lead_scorer._verify_company_fit(
        _company(), _icp(), 0, 0, set(), require_https_transport=True,
    ))
    identity = result.details["dimension_evidence"]["identity"]
    assert result.decision == expected
    assert identity["decision"] == expected
    assert identity["web_identity_receipt"]["reason_code"] == (
        "homepage_web_identity_conflict_unresolved"
        if expected == COMPANY_FIT_UNAVAILABLE else web_reason
    )


def test_verified_alias_and_domain_conflict_keep_existing_classification(monkeypatch):
    async def fetch(_session, _url):
        return 200, "https://qsic.com/", HOMEPAGE

    monkeypatch.setattr(company_verification, "_fetch_bounded_html", fetch)
    homepage = asyncio.run(company_verification.verify_company_exists(
        "Qsic", "https://qsic.com/", company_linkedin="",
        require_https_transport=True,
    ))
    assert not lead_scorer._unresolved_homepage_web_identity_conflict(
        homepage, _web_receipt(reason="verified_rebrand_continuity")
    )
    different_domain = _web_receipt(observed_domain="https://getqsic.com/")
    assert different_domain["decision"] == COMPANY_FIT_MISMATCH
    assert not lead_scorer._unresolved_homepage_web_identity_conflict(
        homepage, different_domain
    )
