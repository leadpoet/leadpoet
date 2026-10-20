"""An exhausted cross-domain lookup can be a negative without a LinkedIn slug.

The identity and final finding shapes reproduce Oct16 Leeward failures:
719c score9 attempt1, homepage4929902, final4930258; 7466 final4930747.
Unresolved source obligations remain retryable. A stale homepage failure can be
superseded only by the existing complete, strict negative investigation proof.
No company-specific acceptance rule or provider response is used by these tests.
"""

import asyncio
import copy

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from lab_arena import scoring
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import competition, lead_scorer as scorer
from qualification.scoring.company_fit_decision import (
    company_fit_match, company_fit_mismatch, company_fit_unavailable,
)


def _identity():
    return {
        "decision": "unavailable", "reason_code": "rebrand_continuity_unproven",
        "submitted_name": "leewardrenewableenergy", "submitted_domain": "leewardenergy.com",
        "submitted_linkedin_slug": "", "observed_name": "leewardrenewableenergy",
        "observed_domain": "lreus.com", "observed_linkedin_slug": "",
        "evidence_source": "company_web_reverification",
    }


def _findings():
    raw = [{
        "target": target, "status": "UNPROVEN", "observed_value": None,
        "evidence_url": "", "evidence_quote": "", "reason": "Completed research found no proof.",
        "old_name": "Leeward Renewable Energy", "new_name": "Leeward Renewable Energy",
        "old_domain": "leewardenergy.com", "new_domain": "lreus.com",
        "shared_linkedin_slug": "",
    } for target in ("stage", "rebrand", "industry")]
    claims = investigator._validated_findings(
        {"findings": raw}, targets=("stage", "rebrand", "industry"),
        fetched_pages={}, fetched_final_urls={}, first_party_domains=set(),
        identity_names={"leewardrenewableenergy"}, identity_anchor=_identity(),
    )
    assert claims is not None
    assert all(value["status"] == "UNPROVEN" for value in claims.values())
    return claims


def _receipt():
    return {
        "gate": "company_evidence_investigation", "completed_submitted_findings": True,
        "failure_reason": "", "targets": ["stage", "rebrand", "industry"],
        "claims": _findings(), "usage": {"fetch_outcomes": [
            {"ok": True, "url_sha256": "a" * 64, "loaded_text_length": 100},
            {"ok": True, "url_sha256": "b" * 64, "loaded_text_length": 100},
        ]},
    }


@pytest.mark.parametrize("observed_slug", ["", "leeward-renewable-energy"])
def test_completed_cross_domain_negative_accepts_explicit_slug(observed_slug):
    identity = {**_identity(), "observed_linkedin_slug": observed_slug}
    assert scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("identity", "industry", "stage"),
        identity_receipt=identity, investigation_receipt=_receipt(),
    )
    # Negative completion does not make this cross-domain company a match.
    assert scorer._web_identity_receipt(
        _company(), _verdict(), verified_rebrand_identity=_findings()["rebrand"],
    )["decision"] == "unavailable"


@pytest.mark.parametrize("fault", [
    "missing_slug", "null_slug", "number_slug", "whitespace_slug", "missing_identity",
    "wrong_source", "missing_name", "missing_receipt", "incomplete", "provider_error",
    "malformed_response", "missing_rebrand", "missing_target", "positive_rebrand",
    "positive_industry", "unvalidated_quote", "profile_failure", "unbound_fault",
    "http_error", "source_blocked", "fetch_failed", "provider_diagnostic_body", "empty_page",
])
def test_cross_domain_negative_keeps_unfinished_proof_retryable(fault):
    identity, receipt = _identity(), _receipt()
    kwargs = {"identity_receipt": identity, "investigation_receipt": receipt}
    if fault == "missing_slug":
        identity.pop("observed_linkedin_slug")
    elif fault in {"null_slug", "number_slug", "whitespace_slug"}:
        identity["observed_linkedin_slug"] = {"null_slug": None, "number_slug": 1, "whitespace_slug": " "}[fault]
    elif fault == "missing_identity":
        kwargs["identity_receipt"] = None
    elif fault == "wrong_source":
        identity["evidence_source"] = "untrusted_model"
    elif fault == "missing_name":
        identity.pop("submitted_name")
    elif fault == "missing_receipt":
        kwargs["investigation_receipt"] = None
    elif fault == "incomplete":
        receipt["completed_submitted_findings"] = False
    elif fault in {"provider_error", "malformed_response"}:
        receipt["failure_reason"] = fault
    elif fault == "missing_rebrand":
        receipt["claims"].pop("rebrand")
    elif fault == "missing_target":
        receipt["targets"].remove("rebrand")
    elif fault in {"positive_rebrand", "positive_industry"}:
        receipt["claims"][fault.removeprefix("positive_")]["status"] = "VERIFIED"
    elif fault == "unvalidated_quote":
        receipt["claims"]["industry"]["evidence_quote"] = "Unvalidated positive proof."
    elif fault == "profile_failure":
        kwargs["linkedin_refresh_outcome"] = "retryable_failure"
    else:
        outcome = {"ok": False, "error_class": "http_error" if fault == "unbound_fault" else fault}
        if fault != "unbound_fault":
            outcome["url_sha256"] = "c" * 64
        receipt["usage"]["fetch_outcomes"].append(outcome)
    assert not scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("identity", "industry", "stage"), **kwargs,
    )


def test_completed_exact_404_is_not_a_transport_failure():
    receipt = _receipt()
    receipt["usage"]["fetch_outcomes"].append({
        "ok": False, "url_sha256": "c" * 64, "error_class": "source_not_found",
    })
    assert scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("identity", "industry", "stage"), identity_receipt=_identity(),
        investigation_receipt=receipt,
    )


def _company():
    return CompanyOutput(
        company_name="Leeward Renewable Energy", company_website="https://leewardenergy.com/",
        company_linkedin="", industry="Energy", employee_count="51-200", country="United States",
        intent_signals=[{
            "description": "Opened an energy facility", "source": "news",
            "url": "https://leewardenergy.com/news", "date": "2026-09-01",
            "snippet": "Leeward opened an energy facility.",
        }],
    )


def _verdict():
    return {
        "observed_company_name": "Leeward Renewable Energy", "observed_company_website": "https://www.lreus.com",
        "observed_company_linkedin": "", "observed_employee_count": "51-200", "employee_size_matches": True,
        "employee_size_evidence_url": "https://example.com/company", "employee_size_evidence_quote": "Leeward Renewable Energy has 51-200 employees.",
        "observed_industry": "Energy", "observed_subindustry": "Renewable power generation",
        "industry_matches": True, "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://www.lreus.com/energy-solutions/", "industry_evidence_quote": "LRE owns and operates wind projects.",
        "observed_hq_country": "United States", "observed_hq_state": "", "geography_matches": True,
        "geography_evidence_url": "https://www.lreus.com/about-us/", "geography_evidence_quote": "Leeward Renewable Energy is headquartered in the United States.",
        "observed_company_stage": "", "stage_matches": None, "stage_evidence_url": "", "stage_evidence_quote": "",
        "reason": "bounded company research",
    }


@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("homepage_fault", [None, "provider_error", "malformed_response"])
@pytest.mark.parametrize("investigation_fault", ["", "provider_error", "malformed_response", "unbound_fault", "incomplete"])
@pytest.mark.parametrize("preserve_marker", [True, False])
def test_negative_completion_through_fit_and_arena(
    monkeypatch, semantic, homepage_fault, investigation_fault, preserve_marker,
):
    company = _company()
    icp = ICPPrompt(
        icp_id="target", prompt="target", industry="Energy", sub_industry="Renewable power generation",
        employee_count="51-200", company_stage="Private Equity", geography="United States",
        country="United States", product_service="power generation",
    )
    async def prechecks(*args, **kwargs):
        return company_fit_match("prechecks passed")
    async def identity(*args, **kwargs):
        if homepage_fault:
            return company_fit_unavailable(
                "website fetch error: ReadError" if homepage_fault == "provider_error" else "website fetch error: homepage response body is unusable",
                details={scorer.VERIFIER_FAILURE_DETAIL_KEY: homepage_fault},
            )
        return company_fit_mismatch("redirect changed registrable domain", details={"identity": {
            **_identity(), "decision": "mismatch", "reason_code": "identity_mismatch",
            "observed_name": "", "evidence_source": "company_homepage",
        }})
    async def request(**kwargs):
        return _verdict(), ""
    async def refresh(verdict, *args, invocation_cache, **kwargs):
        invocation_cache["refresh_outcome"] = "verified"
        return dict(verdict)
    async def investigate(**kwargs):
        receipt = _receipt()
        if investigation_fault == "unbound_fault":
            receipt["usage"]["fetch_outcomes"].append({"ok": False, "error_class": "http_error"})
        return {"claims": receipt["claims"],
                "_completed_submit": investigation_fault != "incomplete",
                "failure_reason": (investigation_fault if investigation_fault in {"provider_error", "malformed_response"} else ""),
                "usage": receipt["usage"]}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "verify_company_exists", identity)
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "_refresh_linkedin_employee_size_observation", refresh)
    monkeypatch.setattr(scorer, "_targeted_company_investigation_dimensions", lambda *a, **k: ("stage", "rebrand", "industry"))
    monkeypatch.setattr(scorer, "_positive_semantic_review_needed", lambda *a: semantic)
    monkeypatch.setattr(scorer, "investigate_company_evidence", investigate)
    if not preserve_marker:
        original = scorer._llm_reverify_company
        async def without_proof_marker(*args, **kwargs):
            result = await original(*args, **kwargs)
            result.details.pop("_completed_investigation_settled", None)
            return result
        monkeypatch.setattr(scorer, "_llm_reverify_company", without_proof_marker)
    fit = asyncio.run(scorer._verify_company_fit(
        company, icp, 0.0, 1.0, set(), require_https_transport=True, evidence_investigator=True,
    ))
    assert fit.decision == "unavailable"
    row = {"final_score": 0.0, "failure_reason": "Company fit unavailable",
           "verifier_gate_receipts": [fit.receipt("company_fit")]}
    retryable = bool(investigation_fault or (homepage_fault and not preserve_marker))
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure(row) is retryable
    assert "_completed_investigation_settled" not in fit.details
    assert row["final_score"] == 0.0
    calls = []
    def judge(*args):
        calls.append(True)
        return [copy.deepcopy(row)]
    def run():
        return scoring.score_work_item(
            {"scored_run_id": "test"}, icp={"employee_count": ["51-200"], "max_companies": 1},
            companies=[{"company_name": company.company_name, "employee_count": "51-200"}], scorer=judge,
        )
    if retryable:
        with pytest.raises(scoring.ScoringError):
            run()
        assert len(calls) == 3
    else:
        assert run()[0]["final_score"] == 0.0
        assert len(calls) == 1


def test_model_fields_cannot_inject_internal_settlement_proof():
    verdict = {**_verdict(), "_completed_investigation_settled": True,
               "failure_class": "insufficient_fit_evidence"}
    result = scorer._reverify_decision(
        verdict, "", "Private Equity", company=_company(),
    )
    assert "_completed_investigation_settled" not in result.details
    assert result.details.get("failure_class") != "insufficient_fit_evidence"
