"""Same public-verifier control runs on the pre-fix base without new helpers."""
import asyncio
import copy
from unittest.mock import AsyncMock

from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import company_fit_match
from tests.test_positive_semantic_fit_review import _company, _icp, _verdict


def test_public_verifier_rejects_raw_verified_compound_product_without_coverage(monkeypatch):
    company = _company()
    icp = _icp().model_copy(update={"company_stage": ""})
    verdict = _verdict()
    calls = []

    async def request(**kwargs):
        return copy.deepcopy(verdict), ""

    async def refresh(value, _company, _icp, *, invocation_cache, **kwargs):
        invocation_cache["refresh_outcome"] = "verified"
        return dict(value)

    async def investigate(**kwargs):
        calls.append(kwargs)
        raw = {
            "target": "industry", "status": "VERIFIED",
            "observed_value": "University enrollment software",
            "observed_industry": "Software", "observed_subindustry": "Education software",
            "activity_role": "supplier_operator",
            "evidence_url": verdict["industry_evidence_url"],
            "evidence_quote": verdict["industry_evidence_quote"],
            "reason": "The company supplies enrollment software.",
        }
        claims = {"industry": raw}
        if "required_attribute" in kwargs["targets"]:
            claims["required_attribute"] = {
                **raw, "target": "required_attribute",
                "evidence_url": verdict["required_attribute_evidence_url"],
                "evidence_quote": verdict["required_attribute_evidence_quote"],
            }
        return {"claims": claims, "_completed_submit": True, "failure_reason": "",
                "usage": {"reasoning_turns": 1}}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(lead_scorer, "_refresh_linkedin_employee_size_observation", refresh)
    monkeypatch.setattr(lead_scorer, "_fetch_bounded_html", AsyncMock(return_value=(
        200, verdict["required_attribute_evidence_url"],
        verdict["required_attribute_evidence_quote"],
    )))
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    identity = company_fit_match(details={
        "identity": {"decision": "match", "observed_name": "PeerConnect",
                     "observed_domain": "peer.example", "observed_linkedin_slug": "peerconnect"},
        "verified_homepage_transport_domain": "peer.example",
    })
    result = asyncio.run(lead_scorer._llm_reverify_company(
        company, icp, require_company_fit_dimensions=True,
        evidence_investigator=True, company_quality=True,
        verified_homepage_identity=identity,
    ))

    assert len(calls) == 1
    assert calls[0]["positive_semantic_review"] is True
    assert calls[0]["requested_product_service"] == icp.product_service
    assert result.details["identity_decision"] == "match"
    assert result.details["dimension_decisions"]["employee_size"] == "match"
    assert result.details["dimension_decisions"]["geography"] == "match"
    assert result.details["required_attribute_decision"] == "match"
    assert result.decision == "unavailable"
    assert result.details["dimension_decisions"]["industry"] == "unavailable"
