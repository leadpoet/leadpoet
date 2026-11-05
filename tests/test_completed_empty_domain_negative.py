"""A completed identity investigation can exhaust an explicitly absent website.

The identity fields below reproduce Nov4 ICP6 Sonar entries 8270605, 8270655,
and 8270705. They are observations, not proof that Owly and Hootsuite match.
The investigation fixture is a validated completed negative with clean source
outcomes. The saved model answers alone do not establish those prerequisites.
"""

import asyncio
import copy

import pytest

from gateway.qualification.models import ICPPrompt
from lab_arena import scoring
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import competition, lead_scorer as scorer
from qualification.scoring.company_fit_decision import (
    company_fit_match, company_fit_mismatch, company_fit_unavailable,
    evaluate_company_identity,
)
from tests.test_completed_cross_domain_negative import (
    _company as complete_company, _identity as complete_identity,
    _receipt as complete_receipt, _verdict as complete_verdict,
)


def _verdict():
    return {
        "observed_company_name": "Owly",
        "observed_company_website": "",
        "observed_company_linkedin": "https://www.linkedin.com/company/owly",
        scorer._REQUIRED_ATTRIBUTE_GROUNDING: {"status": "unproven"},
    }


def _identity(verdict=None):
    verdict = _verdict() if verdict is None else verdict
    return evaluate_company_identity(
        submitted_name="Hootsuite",
        submitted_website="https://ow.ly/h3uR50VFTu9",
        submitted_linkedin="https://www.linkedin.com/company/hootsuite/",
        observed_name=verdict["observed_company_name"],
        observed_website=verdict["observed_company_website"],
        observed_linkedin=verdict["observed_company_linkedin"],
        evidence_source="company_web_reverification",
    )


def _receipt():
    targets = ("rebrand", "industry", "required_attribute")
    findings = investigator._validated_findings(
        {"findings": [{
            "target": target, "status": "UNPROVEN", "observed_value": None,
            "evidence_url": "", "evidence_quote": "",
            "reason": "Completed research found no entity-bound proof.",
        } for target in targets]},
        targets=targets, fetched_pages={}, fetched_final_urls={},
        first_party_domains=set(), identity_names={"hootsuite", "owly"},
        identity_anchor=_identity(),
    )
    assert findings is not None
    return {
        "gate": "company_evidence_investigation",
        "completed_submitted_findings": True, "failure_reason": "",
        "targets": list(targets), "claims": findings,
        "usage": {"fetch_outcomes": [{
            "ok": True, "url_sha256": "a" * 64, "loaded_text_length": 100,
        }]},
    }


def _result():
    return company_fit_unavailable("Identity proof not established", details={
        "identity_decision": "unavailable", "identity_receipt": _identity(),
        "dimension_decisions": {
            "employee_size": "match", "industry": "unavailable",
            "geography": "match", "stage": "match",
        },
        "required_attribute_decision": "unavailable",
        "investigation_receipt": _receipt(),
    })


def _settle(result, verdict, *, profile_cache=None):
    return scorer._settle_completed_company_investigation(
        result, verdict,
        investigation_targets=("rebrand", "industry", "required_attribute"),
        icp=ICPPrompt(
            icp_id="software", prompt="software", industry="Information Technology",
            sub_industry="Software", product_service="Subscription software",
            employee_count="1,001-5,000", company_stage="", geography="Canada",
            country="Canada", required_attribute="Subscription software",
        ),
        icp_attribute="Subscription software", icp_stage="",
        current_profile_cache=profile_cache or {},
    )


def test_saved_empty_website_observation_finishes_as_unavailable_not_a_match():
    result, verdict = _result(), _verdict()
    before = copy.deepcopy(result.details)
    assert before["identity_receipt"]["reason_code"] == "identity_not_proven"
    assert before["identity_receipt"]["observed_domain"] == ""
    settled = _settle(result, verdict)
    assert settled is not None
    assert settled.decision == "unavailable"
    assert settled.details["failure_class"] == "insufficient_fit_evidence"
    assert settled.details["_completed_investigation_settled"] is True
    assert result.details == before
    for key in ("identity_receipt", "identity_decision", "dimension_decisions",
                "required_attribute_decision", "investigation_receipt"):
        assert settled.details[key] == before[key]
    row = {"final_score": 0.0, "failure_reason": "Company fit unavailable",
           "verifier_gate_receipts": [settled.receipt("company_fit")]}
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(row)
    calls = []
    def judge(*_args):
        calls.append(True)
        return [copy.deepcopy(row)]
    scored = scoring.score_work_item(
        {"scored_run_id": "test"},
        icp={"employee_count": ["1,001-5,000"], "max_companies": 1},
        companies=[{"company_name": "Hootsuite", "employee_count": "1,001-5,000"}],
        scorer=judge,
    )
    assert scored[0]["final_score"] == 0.0
    assert calls == [True]


@pytest.mark.parametrize("semantic", [False, True])
def test_empty_domain_negative_survives_the_company_fit_wrapper(monkeypatch, semantic):
    # Only the identity projection is the saved case. Other fit observations
    # are local controls; the completed research still declines qualification.
    company = complete_company().model_copy(update={
        "company_name": "Hootsuite", "company_website": "https://ow.ly/h3uR50VFTu9",
        "company_linkedin": "https://www.linkedin.com/company/hootsuite/",
        "industry": "Information Technology", "employee_count": "1,001-5,000",
        "country": "Canada", "company_stage": "",
    })
    icp = ICPPrompt(
        icp_id="software", prompt="software", industry="Information Technology",
        sub_industry="Software", product_service="Subscription software",
        employee_count="1,001-5,000", company_stage="", geography="Canada",
        country="Canada", required_attribute="Subscription software",
    )
    verdict = {**complete_verdict(), **_verdict(),
        "observed_employee_count": "1,001-5,000", "employee_size_matches": True,
        "observed_hq_country": "Canada", "geography_matches": True,
        "observed_industry": "Information Technology", "industry_matches": False,
        "attribute_satisfied": False,
    }
    requests, investigations = [], []
    async def prechecks(*_args, **_kwargs):
        return company_fit_match("local precheck control")
    async def homepage(*_args, **_kwargs):
        return company_fit_mismatch("cross-domain locator control", details={"identity": {
            **_identity(), "decision": "mismatch", "reason_code": "identity_mismatch",
            "evidence_source": "company_homepage", "observed_name": "",
            "observed_domain": "hootsuite.com",
        }})
    async def request(**_kwargs):
        requests.append(True)
        return copy.deepcopy(verdict), ""
    async def refresh(observation, *_args, invocation_cache, **_kwargs):
        invocation_cache["refresh_outcome"] = "verified"
        return dict(observation)
    async def investigate(**_kwargs):
        investigations.append(True)
        receipt = _receipt()
        return {"claims": receipt["claims"], "_completed_submit": True,
                "failure_reason": "", "usage": receipt["usage"]}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "_refresh_linkedin_employee_size_observation", refresh)
    monkeypatch.setattr(scorer, "_targeted_company_investigation_dimensions",
                        lambda *_a, **_k: ("rebrand", "industry", "required_attribute"))
    monkeypatch.setattr(scorer, "_positive_semantic_review_needed", lambda *_a: semantic)
    monkeypatch.setattr(scorer, "investigate_company_evidence", investigate)
    fit = asyncio.run(scorer._verify_company_fit(
        company, icp, 0.0, 1.0, set(), require_https_transport=True,
        evidence_investigator=True,
    ))
    assert fit.decision == "unavailable"
    assert fit.details["failure_class"] == "insufficient_fit_evidence"
    assert fit.details["company_fit_dimensions"]["identity"] == "unavailable"
    assert "_completed_investigation_settled" not in fit.details
    # The ordinary path retains its existing one schema repair before the
    # investigator. The semantic path reaches the investigator directly.
    assert (len(requests), len(investigations)) == (1 if semantic else 2, 1)
    row = {"final_score": 0.0, "failure_reason": "Company fit unavailable",
           "verifier_gate_receipts": [fit.receipt("company_fit")]}
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(row)


@pytest.mark.parametrize("fault", [
    "missing_website", "null_website", "number_website", "blank_website",
    "invalid_website", "missing_identity", "missing_submitted_domain",
    "missing_observed_name", "missing_observed_slug", "null_observed_slug",
    "blank_observed_slug", "wrong_source", "wrong_reason", "missing_receipt",
    "wrong_gate", "incomplete", "provider_error", "malformed_response",
    "missing_rebrand", "wrong_target", "missing_target", "verified_rebrand",
    "unvalidated_quote", "profile_failure", "fetch_failed", "http_error",
    "provider_diagnostic_body", "empty_page", "unbound_fault",
])
def test_empty_website_keeps_incomplete_or_failed_research_retryable(fault):
    result, verdict = _result(), _verdict()
    identity = result.details["identity_receipt"]
    receipt = result.details["investigation_receipt"]
    profile = {}
    if fault == "missing_website":
        verdict.pop("observed_company_website")
    elif fault in {"null_website", "number_website", "blank_website", "invalid_website"}:
        verdict["observed_company_website"] = {
            "null_website": None, "number_website": 1,
            "blank_website": " ", "invalid_website": "not a URL",
        }[fault]
    elif fault == "missing_identity":
        result.details.pop("identity_receipt")
    elif fault.startswith("missing_") and fault.removeprefix("missing_") in {
        "submitted_domain", "observed_name", "observed_slug",
    }:
        identity.pop(fault.removeprefix("missing_").replace("observed_slug", "observed_linkedin_slug"))
    elif fault in {"null_observed_slug", "blank_observed_slug"}:
        identity["observed_linkedin_slug"] = None if fault.startswith("null") else " "
    elif fault == "wrong_source":
        identity["evidence_source"] = "untrusted_model"
    elif fault == "wrong_reason":
        identity["reason_code"] = "identity_observation_type_invalid"
    elif fault == "missing_receipt":
        result.details.pop("investigation_receipt")
    elif fault == "wrong_gate":
        receipt["gate"] = "untrusted_model"
    elif fault == "incomplete":
        receipt["completed_submitted_findings"] = False
    elif fault in {"provider_error", "malformed_response"}:
        receipt["failure_reason"] = fault
    elif fault == "missing_rebrand":
        receipt["claims"].pop("rebrand")
    elif fault == "wrong_target":
        receipt["claims"]["rebrand"]["target"] = "industry"
    elif fault == "missing_target":
        receipt["targets"].remove("rebrand")
    elif fault == "verified_rebrand":
        receipt["claims"]["rebrand"]["status"] = "VERIFIED"
    elif fault == "unvalidated_quote":
        receipt["claims"]["rebrand"]["evidence_quote"] = "Unvalidated evidence"
    elif fault == "profile_failure":
        profile["refresh_outcome"] = "retryable_failure"
    else:
        outcome = {"ok": False, "error_class": "http_error" if fault == "unbound_fault" else fault}
        if fault != "unbound_fault":
            outcome["url_sha256"] = "b" * 64
        receipt["usage"]["fetch_outcomes"].append(outcome)
    settled = _settle(result, verdict, profile_cache=profile)
    assert settled is None or settled.details.get("failure_class") != "insufficient_fit_evidence"


def test_wrong_entity_remains_a_mismatch():
    verdict = {**_verdict(), "observed_company_website": "https://other-company.example"}
    identity = _identity(verdict)
    assert identity["decision"] == "mismatch"
    assert not scorer._has_explicitly_unproven_fit_dimensions(
        verdict, ("identity",), identity_receipt=identity,
        investigation_receipt=_receipt(),
    )


def test_valid_identity_and_complete_negative_cases_keep_existing_finality():
    result, verdict = _result(), _verdict()
    identity = evaluate_company_identity(
        submitted_name="Hootsuite", submitted_website="https://hootsuite.com",
        submitted_linkedin="https://www.linkedin.com/company/hootsuite/",
        observed_name="Hootsuite", observed_website="https://www.hootsuite.com",
        observed_linkedin="https://www.linkedin.com/company/hootsuite/",
        evidence_source="company_web_reverification",
    )
    assert identity["decision"] == "match"
    result.details["identity_decision"] = "match"
    result.details["identity_receipt"] = identity
    settled = _settle(result, verdict)
    assert settled.decision == "unavailable"
    assert settled.details["identity_decision"] == "match"
    assert settled.details["failure_class"] == "insufficient_fit_evidence"
    assert scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("identity", "industry", "stage"),
        identity_receipt=complete_identity(), investigation_receipt=complete_receipt(),
    )
