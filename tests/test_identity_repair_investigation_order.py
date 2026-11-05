"""Offline reproduction of the Nov5 b66 repair's newly missing geography.

The two saved verifier response shapes are anonymized to Acme. Provider
transports are replaced; native target selection, finding validation,
projection, identity checks and terminal-failure classification still run.
"""
import asyncio
import copy

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import competition, lead_scorer as scorer
from tests.test_company_evidence_investigator import (
    _company, _complete_verdict, _finding, _icp,
)


def _exercise(monkeypatch, *, geography="UNPROVEN", missing_identity=True,
              repair="valid", investigation_failure="", source_fault=False,
              industry="UNPROVEN", initial_rebrand=False):
    company, icp = _company(), _icp(required_attribute="Supplies software.")
    initial = _complete_verdict(
        observed_company_linkedin="" if missing_identity else company.company_linkedin,
        observed_employee_count=None, employee_size_matches=None,
        employee_size_evidence_url="", employee_size_evidence_quote="",
    )
    if initial_rebrand:
        initial.update(observed_company_website="https://other.example",
                       observed_company_linkedin="https://www.linkedin.com/company/other")
    repaired = _complete_verdict(
        observed_employee_count="51-200", employee_size_matches=False,
        employee_size_evidence_quote="Acme has 51-200 employees.",
        observed_hq_country="", observed_hq_state="", geography_matches=None,
        geography_evidence_url="", geography_evidence_quote="",
    )
    for verdict in (initial, repaired):
        verdict.update(attribute_satisfied=True,
                       required_attribute_evidence_url="https://acme.example/about",
                       required_attribute_evidence_quote="Acme supplies SaaS software.")
    if repair == "other_entity":
        repaired.update(observed_company_name="Other", observed_company_website="https://other.example",
                        observed_company_linkedin="https://www.linkedin.com/company/other")
    elif repair == "missing":
        repaired["observed_company_linkedin"] = ""
    events, investigations = [], []

    async def request(**kwargs):
        events.append(kwargs["telemetry_purpose"])
        if len(events) == 1:
            return copy.deepcopy(initial), ""
        if repair in {"provider_error", "malformed_response"}:
            kwargs["diagnostic"]["failure_reason_code"] = repair
            return None, repair
        return copy.deepcopy(repaired), ""

    async def refresh(verdict, *_args, **_kwargs):
        return verdict

    async def no_redirect(_company, verdict):
        return verdict

    async def investigate(**kwargs):
        events.append("investigate")
        investigations.append(kwargs)
        url = "https://acme.example/about"
        text = ("Acme has 11-50 employees. "
                "Acme is headquartered in California, United States. "
                "Acme supplies SaaS software.")
        findings = []
        for target in kwargs["targets"]:
            if target == "headcount":
                findings.append(_finding(target, observed_value="11-50", evidence_url=url,
                                         evidence_quote="Acme has 11-50 employees."))
            elif target == "geography" and geography != "UNPROVEN":
                findings.append(_finding(target, status=geography, observed_value="United States",
                                         observed_country="United States", observed_state="California",
                                         evidence_url=url,
                                         evidence_quote="Acme is headquartered in California, United States."))
            elif target == "industry" and industry == "VERIFIED":
                findings.append(_finding(target, observed_industry="Software", observed_subindustry="SaaS",
                                         activity_role="supplier_operator", evidence_url=url,
                                         evidence_quote="Acme supplies SaaS software."))
            elif target == "required_attribute":
                findings.append(_finding(target, activity_role="supplier_operator", evidence_url=url,
                                         evidence_quote="Acme supplies SaaS software."))
            else:
                findings.append(_finding(target, status="UNPROVEN", observed_value=None,
                                         evidence_url="", evidence_quote=""))
        claims = investigator._validated_findings(
            {"findings": findings}, targets=kwargs["targets"], fetched_pages={url: text},
            fetched_final_urls={url: url}, first_party_domains={"acme.example"}, identity_names={"acme"},
            identity_anchor={"submitted_domain": "acme.example", "observed_domain": "acme.example"},
        )
        return {
            "claims": claims, "_completed_submit": not investigation_failure,
            "failure_reason": investigation_failure,
            "usage": {"search_calls": 0, "fetch_calls": 1,
                      "fetch_outcomes": [{"url_sha256": "a" * 64, "ok": not source_fault,
                                          "loaded_text_length": len(text) if not source_fault else 0,
                                          **({"error_class": "http_error"} if source_fault else {})}]},
        }

    class OfflineSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        def get(self, *_args, **_kwargs):
            raise AssertionError("offline regression must not contact providers")

        post = get

    async def fetch_page(_session, url, **_kwargs):
        assert url == "https://acme.example/about"
        return 200, url, "Acme supplies SaaS software."

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "offline-test-model")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "_refresh_linkedin_employee_size_observation", refresh)
    monkeypatch.setattr(scorer, "_resolve_observed_linkedin_redirect_alias", no_redirect)
    monkeypatch.setattr(scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(scorer, "_fetch_bounded_html", fetch_page)
    monkeypatch.setattr(scorer.aiohttp, "ClientSession", OfflineSession)
    result = asyncio.run(scorer._llm_reverify_company(
        company, icp, require_company_fit_dimensions=True,
        company_quality=True, evidence_investigator=True,
    ))
    return result, events, investigations


@pytest.mark.parametrize("geography", ["UNPROVEN", "VERIFIED"])
def test_identity_repair_precedes_one_investigation_of_final_gaps(monkeypatch, geography):
    result, events, investigations = _exercise(monkeypatch, geography=geography)
    assert events == ["lead_scorer_reverify", "lead_scorer_reverify_schema_repair", "investigate"]
    assert len(investigations) == 1
    assert set(investigations[0]["targets"]) == {"headcount", "industry", "geography", "required_attribute"}
    assert result.details["identity_decision"] == "match"
    assert result.details["dimension_decisions"]["employee_size"] == "match"
    assert result.details["dimension_decisions"]["industry"] == "unavailable"
    assert result.details["dimension_decisions"]["geography"] == (
        "match" if geography == "VERIFIED" else "unavailable"
    )
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert result.details["required_attribute_decision"] == "match"
    assert result.decision == "unavailable"
    fit = scorer._complete_company_fit_result(
        result.decision, result.reason, dimensions=result.details["dimension_decisions"],
        dimension_evidence=result.details["dimension_evidence"], stage_required=False,
        supporting_receipts=[result.details["investigation_receipt"]],
        failure_class=result.details["failure_class"],
    )
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure({
        "final_score": 0.0, "verifier_gate_receipts": [fit.receipt("company_fit")],
    })


def test_valid_identity_keeps_original_one_investigation_without_schema_repair(monkeypatch):
    result, events, investigations = _exercise(monkeypatch, missing_identity=False)
    assert events == ["lead_scorer_reverify", "investigate"]
    assert len(investigations) == 1
    assert "geography" not in investigations[0]["targets"]
    assert result.details["failure_class"] == "insufficient_fit_evidence"


def test_all_verified_repaired_dimensions_can_match(monkeypatch):
    result, events, investigations = _exercise(monkeypatch, geography="VERIFIED", industry="VERIFIED")
    assert events == ["lead_scorer_reverify", "lead_scorer_reverify_schema_repair", "investigate"]
    assert len(investigations) == 1
    assert result.decision == "match"


def test_existing_rebrand_investigation_is_not_deferred(monkeypatch):
    result, events, investigations = _exercise(monkeypatch, initial_rebrand=True)
    assert events == ["lead_scorer_reverify", "investigate"]
    assert len(investigations) == 1
    assert "rebrand" in investigations[0]["targets"]
    assert result.decision != "match"


@pytest.mark.parametrize("repair", ["other_entity", "missing", "provider_error", "malformed_response"])
def test_unresolved_or_changed_identity_cannot_qualify(monkeypatch, repair):
    result, _events, investigations = _exercise(monkeypatch, repair=repair)
    assert len(investigations) <= 1
    assert result.decision != "match"
    assert result.details.get("identity_decision") != "match"
    if repair in {"provider_error", "malformed_response"}:
        assert result.details.get("failure_class") != "insufficient_fit_evidence"
        assert competition.scorer_breakdown_has_retryable_infrastructure_failure({
            "verifier_gate_receipts": [result.receipt("company_fit")],
        })


@pytest.mark.parametrize("failure,source_fault", [("provider_error", False), ("", True)])
def test_post_repair_investigation_keeps_provider_faults_retryable(monkeypatch, failure, source_fault):
    result, _events, investigations = _exercise(
        monkeypatch, investigation_failure=failure, source_fault=source_fault,
    )
    assert len(investigations) == 1
    assert result.decision != "match"
    assert result.details.get("failure_class") != "insufficient_fit_evidence"
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    })
