"""A first-party entity must survive an unbound web identity claim."""

import asyncio
import copy

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_verification, competition, lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
    company_fit_unavailable,
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
        (HOMEPAGE, 503, False),
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


def _completed_qsic_review(identity=None):
    # Mirrors the completed shadow review: the submitted qsic.com identity
    # conflicts with the independently bound first-party organization, while
    # the model's Series B/headcount citations describe retail-media QSIC.
    identity = identity or {
        **_web_receipt(),
        "decision": COMPANY_FIT_UNAVAILABLE,
        "reason_code": "homepage_web_identity_conflict_unresolved",
    }
    claims = {
        target: {
            "target": target, "status": "UNPROVEN",
            "evidence_url": "", "evidence_quote": "",
        }
        for target in ("rebrand", "industry", "required_attribute")
    }
    claims["stage"] = {
        "target": "stage", "status": "VERIFIED",
        "evidence_url": "https://www.getqsic.com/post/qsic-seriesb-25million",
        "evidence_quote": "QSIC announced its Series B.",
    }
    claims["headcount"] = {
        "target": "headcount", "status": "VERIFIED",
        "evidence_url": "https://www.linkedin.com/company/qsic",
        "evidence_quote": "QSIC has 51-200 employees.",
    }
    receipt = {
        "gate": "company_evidence_investigation",
        "completed_submitted_findings": True,
        "failure_reason": "",
        "targets": ["stage", "rebrand", "headcount", "industry", "required_attribute"],
        "claims": claims,
        "usage": {"fetch_outcomes": [
            {"ok": True, "url_sha256": "a" * 64, "loaded_text_length": 100},
        ]},
    }
    result = company_fit_unavailable("unproven dimensions", details={
        "identity_decision": COMPANY_FIT_UNAVAILABLE,
        "identity_receipt": identity,
        "dimension_decisions": {
            "identity": COMPANY_FIT_UNAVAILABLE,
            "employee_size": COMPANY_FIT_MATCH,
            "industry": COMPANY_FIT_UNAVAILABLE,
            "geography": COMPANY_FIT_MATCH,
            "stage": COMPANY_FIT_MATCH,
        },
        "required_attribute_decision": COMPANY_FIT_UNAVAILABLE,
        "investigation_receipt": receipt,
    })
    icp = _icp().model_copy(update={
        "company_stage": "Series B", "required_attribute": "Retail media",
    })
    verdict = {lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING: {"status": "unproven"}}
    return result, verdict, icp


def _settle_qsic(result, verdict, icp, *, refresh_outcome=""):
    return lead_scorer._settle_completed_company_investigation(
        result, verdict,
        investigation_targets=("stage", "rebrand", "headcount", "industry", "required_attribute"),
        icp=icp, icp_attribute=icp.required_attribute,
        icp_stage=icp.company_stage,
        current_profile_cache={"refresh_outcome": refresh_outcome},
    )


def test_completed_qsic_name_conflict_is_a_terminal_unavailable_zero():
    result, verdict, icp = _completed_qsic_review()
    assert "rebrand" in lead_scorer._targeted_company_investigation_dimensions(
        result, icp_stage=icp.company_stage, employee_size_conflict=False,
        company=_company(),
    )
    settled = _settle_qsic(result, verdict, icp)
    assert settled is not None
    assert settled.decision == COMPANY_FIT_UNAVAILABLE
    assert settled.details["identity_decision"] == COMPANY_FIT_UNAVAILABLE
    assert settled.details["failure_class"] == "insufficient_fit_evidence"
    row = {"final_score": 0.0, "failure_reason": "Company fit unavailable",
           "verifier_gate_receipts": [settled.receipt("company_fit")]}
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(row)


@pytest.mark.parametrize("fault", [
    "incomplete", "provider_error", "fetch_failed", "linkedin_refresh",
    "missing_bound_homepage", "missing_rebrand_finding", "verified_rebrand_claim",
    "uninvestigated_dimension",
])
def test_qsic_finality_keeps_unfinished_or_unbound_cases_retryable(fault):
    result, verdict, icp = _completed_qsic_review()
    result = copy.deepcopy(result)
    details = result.details
    receipt = details["investigation_receipt"]
    if fault == "incomplete":
        receipt["completed_submitted_findings"] = False
    elif fault == "provider_error":
        receipt["failure_reason"] = "provider_error"
    elif fault == "fetch_failed":
        receipt["usage"]["fetch_outcomes"].append({
            "ok": False, "url_sha256": "b" * 64, "error_class": "http_error",
        })
    elif fault == "missing_bound_homepage":
        details["identity_receipt"]["evidence_source"] = ""
    elif fault == "missing_rebrand_finding":
        receipt["claims"].pop("rebrand")
    elif fault == "verified_rebrand_claim":
        receipt["claims"]["rebrand"]["status"] = "VERIFIED"
    elif fault == "uninvestigated_dimension":
        details["dimension_decisions"]["geography"] = COMPANY_FIT_UNAVAILABLE
    settled = _settle_qsic(
        result, verdict, icp,
        refresh_outcome="retryable_failure" if fault == "linkedin_refresh" else "",
    )
    assert settled is None or settled.details.get("failure_class") != "insufficient_fit_evidence"


def test_valid_identity_keeps_existing_completed_negative_path():
    result, verdict, icp = _completed_qsic_review()
    result.details["identity_decision"] = COMPANY_FIT_MATCH
    result.details["dimension_decisions"]["identity"] = COMPANY_FIT_MATCH
    result.details["identity_receipt"]["decision"] = COMPANY_FIT_MATCH
    settled = _settle_qsic(result, verdict, icp)
    assert settled is not None
    assert settled.decision == COMPANY_FIT_UNAVAILABLE
    assert settled.details["failure_class"] == "insufficient_fit_evidence"
