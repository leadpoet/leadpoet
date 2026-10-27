"""A failed optional profile cannot reopen a completed negative fit review."""
from __future__ import annotations

import asyncio
import copy

import pytest

from qualification.scoring import lead_scorer as scorer
from qualification.scoring.company_fit_decision import company_fit_unavailable
from tests.test_current_linkedin_company_size import _icp


def _case(control=None):
    icp = _icp().model_copy(update={"required_attribute": "Provides workflow software"})
    claims = {
        target: {"target": target, "status": "UNPROVEN", "evidence_url": "", "evidence_quote": ""}
        for target in ("industry", "required_attribute")
    }
    claims["headcount"] = {
        "target": "headcount", "status": "VERIFIED", "observed_value": "51-200",
        "evidence_url": "https://www.linkedin.com/company/acme",
        "evidence_quote": "Acme | LinkedIn. Company size 51-200 employees",
    }
    receipt = {
        "gate": "company_evidence_investigation", "completed_submitted_findings": True,
        "failure_reason": "", "targets": list(claims), "claims": claims,
        "usage": {"fetch_outcomes": [{"ok": True, "url_sha256": "a" * 64, "loaded_text_length": 100}]},
    }
    result = company_fit_unavailable("unproven dimensions: industry, required_attribute", details={
        "identity_decision": "match",
        "identity_receipt": {"decision": "match", "evidence_source": "company_web_reverification"},
        "dimension_decisions": {"employee_size": "match", "industry": "unavailable", "geography": "match", "stage": "match"},
        "required_attribute_decision": "unavailable", "investigation_receipt": receipt,
    })
    # Decision results own their receipt copy. Mutate that copy for controls.
    receipt = result.details["investigation_receipt"]
    claims = receipt["claims"]
    verdict = {scorer._REQUIRED_ATTRIBUTE_GROUNDING: {"status": "unproven"}}
    cache = {
        "refresh_outcome": "retryable_failure", "structured_failure": True,
        "structured_failure_reason": "malformed_response", "failure_reason_code": "malformed_response",
        "profile_url": "https://www.linkedin.com/company/acme-old",
        "evidence": {"outcome": "insufficient_evidence"},
    }
    if control in {"provider_error", "source_blocked"}:
        cache["structured_failure_reason"] = cache["failure_reason_code"] = control
    elif control == "initial_missing":
        cache["evidence"] = None
    elif control == "unknown_current":
        cache["evidence"]["outcome"] = "unknown"
    elif control == "incomplete":
        receipt["completed_submitted_findings"] = False
    elif control == "review_failure":
        receipt["failure_reason"] = "provider_error"
    elif control in {"transport", "unbound_transport"}:
        receipt["usage"]["fetch_outcomes"].append({
            "ok": False, "error_class": "http_error",
            **({"url_sha256": "b" * 64} if control == "transport" else {}),
        })
    elif control == "empty_text":
        receipt["usage"]["fetch_outcomes"][0]["loaded_text_length"] = 0
    elif control == "no_sources":
        receipt["usage"]["fetch_outcomes"] = []
    elif control == "unbound_source":
        receipt["usage"]["fetch_outcomes"][0].pop("url_sha256")
    elif control == "missing_claim":
        claims.pop("industry")
    elif control == "unresolved_positive":
        claims["industry"]["status"] = "VERIFIED"
    elif control == "cited_unproven":
        claims["industry"]["evidence_url"] = "https://acme.example.com/about"
    elif control == "attribute_grounding":
        verdict[scorer._REQUIRED_ATTRIBUTE_GROUNDING]["status"] = "failed"
    elif control == "unbound_identity":
        result.details["identity_decision"] = "unavailable"
        result.details["identity_receipt"] = {"decision": "unavailable", "evidence_source": ""}
        receipt["targets"].append("rebrand")
        claims["rebrand"] = {"target": "rebrand", "status": "UNPROVEN", "evidence_url": "", "evidence_quote": ""}
    elif control == "uninvestigated_dimension":
        result.details["dimension_decisions"]["geography"] = "unavailable"
    return result, verdict, icp, cache


@pytest.mark.parametrize("control", [
    None, "provider_error", "source_blocked", "initial_missing", "unknown_current",
    "incomplete", "review_failure", "transport", "unbound_transport", "empty_text",
    "no_sources", "unbound_source", "missing_claim", "unresolved_positive", "cited_unproven",
    "attribute_grounding", "unbound_identity", "uninvestigated_dimension",
])
def test_completed_negative_review_after_recovered_headcount_keeps_fault_guards(control):
    result, verdict, icp, cache = _case(control)
    before = copy.deepcopy(cache)
    dimensions_before = copy.deepcopy(result.details["dimension_decisions"])
    settled = scorer._settle_completed_company_investigation(
        result, verdict, investigation_targets=tuple(result.details["investigation_receipt"]["targets"]),
        icp=icp, icp_attribute=icp.required_attribute, icp_stage=icp.company_stage,
        current_profile_cache=cache,
    )
    assert cache == before
    assert settled is None or settled.decision == "unavailable"
    assert settled is None or settled.details["dimension_decisions"] == dimensions_before
    terminal = settled is not None and settled.details.get("failure_class") == "insufficient_fit_evidence"
    assert terminal is (control is None)
    if terminal:
        assert settled.details["investigation_receipt"]["prior_linkedin_refresh_failure_reason"] == "malformed_response"


@pytest.mark.parametrize("phase", ["initial", "post_repair"])
@pytest.mark.parametrize("semantic_review", [False, True])
def test_recovered_headcount_negative_finality_through_each_investigation_path(monkeypatch, phase, semantic_review):
    from gateway.qualification.models import ICPPrompt
    from qualification.scoring import company_evidence_investigator as investigator, competition
    from tests.test_semantic_review_infrastructure_guard import _company, _verdict

    company = _company()
    icp = ICPPrompt(
        icp_id="recovered-review", prompt="target", industry="Software", sub_industry="SaaS",
        employee_count="51-200", company_stage="Series A", geography="United States",
        country="United States", product_service="software",
    )
    requests, target_calls, semantic_calls, investigations, caches = [], [], [], [], []

    async def request(**kwargs):
        requests.append(kwargs["telemetry_purpose"])
        return _verdict(), ""

    async def refresh(verdict, _company, _icp, *, invocation_cache, **kwargs):
        invocation_cache.update(
            refresh_outcome="retryable_failure", structured_failure=True,
            structured_failure_reason="malformed_response", failure_reason_code="malformed_response",
            profile_url="https://www.linkedin.com/company/acme-old",
            evidence={"outcome": "insufficient_evidence"},
        )
        caches.append(invocation_cache)
        return dict(verdict)

    def targets(*args, **kwargs):
        target_calls.append(True)
        return () if phase == "post_repair" and len(target_calls) == 1 else ("headcount", "industry", "geography", "stage")

    def semantic(*args):
        semantic_calls.append(True)
        return semantic_review and (phase == "initial" or len(semantic_calls) > 1)

    async def investigate(**kwargs):
        investigations.append(kwargs)
        url = "https://www.linkedin.com/company/acme"
        quote = "Acme | LinkedIn. Company size 51-200 employees"
        headcount = investigator._validated_findings(
            {"findings": [{"target": "headcount", "status": "VERIFIED", "observed_value": "51-200",
                           "evidence_url": url, "evidence_quote": quote}]},
            targets=("headcount",), fetched_pages={url: quote}, fetched_final_urls={url: url},
            first_party_domains={"acme.example"}, identity_names={"acme"},
        )["headcount"]
        assert headcount["status"] == "VERIFIED"
        claims = {target: {"target": target, "status": "UNPROVEN", "evidence_url": "", "evidence_quote": ""}
                  for target in kwargs["targets"]}
        claims["headcount"] = headcount
        return {
            "claims": claims, "_completed_submit": True, "failure_reason": "",
            investigator.PRIVATE_FETCHED_PAGES_KEY: {url: {"final_url": url, "text": quote}},
            "usage": {"fetch_outcomes": [{"ok": True, "url_sha256": "a" * 64, "loaded_text_length": len(quote)}]},
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "_refresh_linkedin_employee_size_observation", refresh)
    monkeypatch.setattr(scorer, "_targeted_company_investigation_dimensions", targets)
    monkeypatch.setattr(scorer, "_positive_semantic_review_needed", semantic)
    monkeypatch.setattr(scorer, "investigate_company_evidence", investigate)
    result = asyncio.run(scorer._llm_reverify_company(
        company, icp, require_company_fit_dimensions=True, evidence_investigator=True,
    ))
    assert len(investigations) == 1
    assert len(requests) == (1 if phase == "initial" else 2)
    assert investigations[0]["positive_semantic_review"] is semantic_review
    assert result.decision == "unavailable"
    assert result.details["dimension_decisions"]["employee_size"] == "match"
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert all(cache["refresh_outcome"] == "retryable_failure" and cache["failure_reason_code"] == "malformed_response" for cache in caches)
    fit = scorer._complete_company_fit_result(
        result.decision, result.reason, dimensions=result.details["dimension_decisions"],
        dimension_evidence=result.details["dimension_evidence"], stage_required=True,
        supporting_receipts=[result.details["investigation_receipt"]], failure_class=result.details["failure_class"],
    )
    row = {"final_score": 0.0, "failure_reason": "Company fit unavailable", "verifier_gate_receipts": [fit.receipt("company_fit")]}
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(row)
