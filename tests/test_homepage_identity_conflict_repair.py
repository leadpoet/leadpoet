"""Frozen Nov20 Dalfen identity, with independent complete-fit test controls.

The 39670ecb verifier returned the current slug while the bound homepage
retained the old slug. The exact old profile completed HTTP 200 with
error=null, element=null. Neither observation proves that the slugs alias.
Only provider transports and the already completed fit investigation are
replaced here; identity orchestration and pipeline finality remain native.
"""
import asyncio
from copy import deepcopy

import pytest

from qualification.scoring import competition, lead_scorer as scorer
from qualification.scoring.company_fit_decision import company_fit_match, company_fit_unavailable
from lab_arena import company_judgments
from tests.test_fit_evidence_url_compat import _company, _complete_verdict, _icp


OLD = "https://www.linkedin.com/company/dalfen-america-corp"
CURRENT = "https://www.linkedin.com/company/dalfen-industrial"


def _saved_homepage():
    return company_fit_unavailable("homepage identity evidence unavailable", details={
        "verified_homepage_transport_domain": "dalfen.com",
        "observed_homepage_organization_identity": {
            "normalized_name": "Dalfen Industrial",
            "registrable_dns_domain": "dalfen.com",
            "linkedin_company_slug": "dalfen-america-corp",
        },
        "identity": {
            "decision": "unavailable", "reason_code": "identity_not_proven",
            "evidence_source": "company_homepage",
            "submitted_name": "dalfenindustrial", "observed_name": "dalfenindustrial",
            "submitted_domain": "dalfen.com", "observed_domain": "dalfen.com",
            "submitted_linkedin_slug": "dalfen-industrial",
            "observed_linkedin_slug": "dalfen-america-corp",
        },
    })


def _exercise(monkeypatch, *, repair="same", profile="empty", homepage_fault=False,
              investigation=False, source_fault=False):
    company = _company(company_name="Dalfen Industrial", company_website="https://www.dalfen.com/",
                       company_linkedin=CURRENT, company_stage="", state="Texas")
    icp = _icp().model_copy(update={"company_stage": ""})
    verdict = _complete_verdict(name="Dalfen Industrial", website="https://www.dalfen.com/")
    verdict.update(observed_company_linkedin=CURRENT, observed_hq_state="Texas",
                   geography_evidence_quote="Dalfen Industrial is headquartered in Texas, United States.")
    home = _saved_homepage()
    if homepage_fault:
        home = company_fit_unavailable("website returned HTTP 503", details=home.details)
    if profile == "matching":
        home.details["identity"]["observed_linkedin_slug"] = "dalfen-industrial"
        home.details["observed_homepage_organization_identity"]["linkedin_company_slug"] = "dalfen-industrial"
    events, prompts, targets = [], [], []

    async def request(**kwargs):
        purpose = kwargs["telemetry_purpose"]
        events.append(purpose)
        prompts.append(kwargs["prompt"])
        value = deepcopy(verdict)
        if purpose.endswith("schema_repair"):
            if repair in {"provider_error", "malformed_response"}:
                kwargs["diagnostic"][scorer.VERIFIER_FAILURE_REASON_KEY] = repair
                return None, repair
            if repair == "empty":
                value["observed_company_linkedin"] = ""
            if repair == "null":
                value["observed_company_linkedin"] = None
        return value, ""

    async def passthrough(value, *args, **kwargs):
        return dict(value)

    async def attribute(value, **kwargs):
        return dict(value), None

    async def structured(domain, url, **kwargs):
        events.append("old_profile")
        assert domain == "dalfen.com" and url == OLD
        if profile == "fault":
            kwargs["diagnostic"][scorer.VERIFIER_FAILURE_REASON_KEY] = "provider_error"
        if profile == "canonical":
            kwargs["company_identity_evidence"].update({
                "name": "Dalfen Industrial", "website": "https://www.dalfen.com/",
                "url": CURRENT, "requested_url": OLD, "company_id": "12345",
                "provider": scorer.STRUCTURED_PROFILE_PROVIDER,
                "source_field": scorer.STRUCTURED_PROFILE_IDENTITY_SOURCE_FIELD,
            })
        return None

    async def investigate(**kwargs):
        events.append("investigate")
        targets.append(tuple(kwargs["investigation_targets"]))
        result = deepcopy(kwargs["prior_result"])
        claims = {target: {"target": target, "status": "VERIFIED"}
                  for target in kwargs["investigation_targets"]}
        result.details["investigation_receipt"] = {
            "gate": "company_evidence_investigation", "completed_submitted_findings": True,
            "failure_reason": "", "targets": list(claims), "claims": claims,
            "usage": {"fetch_outcomes": [{"url_sha256": "a" * 64, "ok": not source_fault,
                "loaded_text_length": 0 if source_fault else 200,
                **({"error_class": "http_error"} if source_fault else {})}]},
        }
        return dict(kwargs["verdict"]), result, claims, {}, {}

    async def identity(*args, **kwargs):
        return home

    async def precheck(*args, **kwargs):
        return company_fit_match()

    async def forbidden(*args, **kwargs):
        pytest.fail("No additional transport or redirect research is authorized by this repair")

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "_ground_required_attribute_evidence", attribute)
    monkeypatch.setattr(scorer, "_ground_employee_size_observation", passthrough)
    monkeypatch.setattr(scorer, "_refresh_linkedin_employee_size_observation", passthrough)
    monkeypatch.setattr(scorer, "fetch_structured_linkedin_company_size", structured)
    monkeypatch.setattr(scorer, "_run_targeted_company_evidence_investigation", investigate)
    if investigation:
        monkeypatch.setattr(scorer, "_positive_semantic_review_needed", lambda *args: True)
    monkeypatch.setattr(scorer, "verify_company_exists", identity)
    monkeypatch.setattr(scorer, "run_company_zero_checks", precheck)
    monkeypatch.setattr(scorer, "_fetch_bounded_html", forbidden)
    fit = asyncio.run(scorer._verify_company_fit(
        company, icp, 0, 1, set(), require_https_transport=True,
        company_quality=True, evidence_investigator=investigation,
    ))
    return fit, events, prompts, targets


@pytest.mark.parametrize("investigation", [False, True])
def test_saved_dalfen_conflict_uses_one_identity_repair_and_finishes_unqualified(monkeypatch, investigation):
    fit, events, prompts, targets = _exercise(monkeypatch, investigation=investigation)
    assert events == ["lead_scorer_reverify", "old_profile", "lead_scorer_reverify_schema_repair"] + (
        ["investigate"] if investigation else [])
    assert "identity" in prompts[1].split("SCHEMA REPAIR:")[1]
    assert "dalfen-america-corp" in prompts[1] and "dalfen-industrial" in prompts[1]
    assert all("rebrand" not in target for target in targets)
    assert fit.decision == "unavailable"
    assert fit.details["company_fit_dimensions"]["identity"] == "unavailable"
    assert fit.details["failure_class"] == "insufficient_fit_evidence"
    receipt = fit.details["dimension_evidence"]["identity"]["web_identity_receipt"]
    assert receipt["observed_linkedin_slug"] == "dalfen-industrial"
    assert receipt["homepage_identity"]["observed_linkedin_slug"] == "dalfen-america-corp"
    raw = {
        "company_judgment_policy": "company_judgment_v1", "final_score": 0.0,
        "company_identity_key": "domain:dalfen.com",
        "company_identity_alias_keys": ["domain:dalfen.com"],
        "verifier_gate_receipts": [fit.receipt("company_fit")], "intent_signals_detail": [],
    }
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(raw)
    assert company_judgments.raw_judgment_is_cacheable(raw)


@pytest.mark.parametrize("profile", ["canonical", "matching"])
def test_proven_identity_has_no_extra_review_or_calls(monkeypatch, profile):
    fit, events, _, _ = _exercise(monkeypatch, profile=profile)
    assert fit.decision == "match"
    assert events == ["lead_scorer_reverify"] + (["old_profile"] if profile == "canonical" else [])


@pytest.mark.parametrize("kwargs", [
    {"repair": "provider_error"}, {"repair": "malformed_response"},
    {"repair": "null"}, {"profile": "fault"}, {"homepage_fault": True},
    {"investigation": True, "source_fault": True},
])
def test_unfinished_identity_or_source_fault_stays_retryable(monkeypatch, kwargs):
    fit, events, _, _ = _exercise(monkeypatch, **kwargs)
    assert events.count("lead_scorer_reverify_schema_repair") == 1
    assert fit.decision == "unavailable"
    assert fit.details.get("failure_class") != "insufficient_fit_evidence"
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure({
        "final_score": 0.0, "verifier_gate_receipts": [fit.receipt("company_fit")],
    })


def test_explicit_empty_identity_after_healthy_review_retains_existing_finality(monkeypatch):
    fit, events, _, _ = _exercise(monkeypatch, repair="empty")
    assert events.count("lead_scorer_reverify_schema_repair") == 1
    assert fit.decision == "unavailable"
    assert fit.details["failure_class"] == "insufficient_fit_evidence"
