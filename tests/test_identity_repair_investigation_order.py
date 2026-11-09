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
              industry="UNPROVEN", initial_rebrand=False, company_quality=True,
              initial_missing_name=False, profile_failure="", initial_missing_geography=False):
    company, icp = _company(), _icp(required_attribute="Supplies software.")
    initial = _complete_verdict(
        observed_company_linkedin="" if missing_identity else company.company_linkedin,
        observed_employee_count=None, employee_size_matches=None,
        employee_size_evidence_url="", employee_size_evidence_quote="",
    )
    if initial_rebrand:
        initial.update(observed_company_website="https://other.example",
                       observed_company_linkedin="https://www.linkedin.com/company/other")
    if initial_missing_geography:
        initial.update(observed_hq_country="", observed_hq_state="", geography_matches=None,
                       geography_evidence_url="", geography_evidence_quote="")
    if initial_missing_name:
        initial.pop("observed_company_name")
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
    elif repair in {"omitted", "null", "number", "whitespace", "invalid"}:
        repaired["observed_company_linkedin"] = {
            "omitted": "", "null": None, "number": 3,
            "whitespace": " ", "invalid": "not a LinkedIn URL",
        }[repair]
        if repair == "omitted":
            repaired.pop("observed_company_linkedin")
    elif repair in {"wrong_name", "wrong_domain"}:
        repaired["observed_company_linkedin"] = ""
        repaired["observed_company_name" if repair == "wrong_name"
                 else "observed_company_website"] = (
            "Other" if repair == "wrong_name" else "https://other.example"
        )
    events, investigations = [], []

    async def request(**kwargs):
        events.append(kwargs["telemetry_purpose"])
        if len(events) == 1:
            return copy.deepcopy(initial), ""
        if repair in {"provider_error", "malformed_response"}:
            kwargs["diagnostic"]["failure_reason_code"] = repair
            return None, repair
        return copy.deepcopy(repaired), ""

    async def refresh(verdict, *_args, **kwargs):
        if profile_failure:
            kwargs["invocation_cache"].update(
                refresh_outcome="retryable_failure",
                failure_reason_code=profile_failure,
            )
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
        company_quality=company_quality, evidence_investigator=True,
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


@pytest.mark.parametrize("geography,industry", [
    ("VERIFIED", "VERIFIED"), ("UNPROVEN", "UNPROVEN"),
])
@pytest.mark.parametrize("initial_missing_name", [False, True])
@pytest.mark.parametrize("repair", ["missing", "wrong_name"])
def test_explicit_identity_repair_can_exhaust_missing_linkedin(
    monkeypatch, geography, industry, initial_missing_name, repair,
):
    result, events, investigations = _exercise(
        monkeypatch, repair=repair, geography=geography, industry=industry,
        initial_missing_name=initial_missing_name,
    )
    assert events == ["lead_scorer_reverify", "lead_scorer_reverify_schema_repair", "investigate"]
    assert len(investigations) == 1
    assert "rebrand" not in investigations[0]["targets"]
    assert result.decision == "unavailable"
    assert result.details["identity_decision"] == "unavailable"
    assert result.details["identity_receipt"]["reason_code"] == "identity_not_proven"
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    })


@pytest.mark.parametrize("repair", [
    "omitted", "null", "number", "whitespace", "invalid",
    "provider_error", "malformed_response",
])
def test_only_explicit_typed_identity_repair_can_exhaust(monkeypatch, repair):
    result, events, investigations = _exercise(monkeypatch, repair=repair)
    assert events.count("lead_scorer_reverify_schema_repair") == 1
    assert len(investigations) <= 1
    assert result.decision != "match"
    assert result.details.get("failure_class") != "insufficient_fit_evidence"
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    })


@pytest.mark.parametrize("failure,source_fault,profile_failure", [
    ("provider_error", False, ""), ("malformed_response", False, ""),
    ("", True, ""), ("", False, "provider_error"),
])
def test_missing_linkedin_repair_does_not_clear_other_faults(
    monkeypatch, failure, source_fault, profile_failure,
):
    result, _events, _investigations = _exercise(
        monkeypatch, repair="missing", investigation_failure=failure,
        source_fault=source_fault, profile_failure=profile_failure,
    )
    assert result.decision != "match"
    assert result.details.get("failure_class") != "insufficient_fit_evidence"
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure({
        "verifier_gate_receipts": [result.receipt("company_fit")],
    })


def test_unrelated_completed_investigation_cannot_exhaust_missing_identity(monkeypatch):
    result, _events, _investigations = _exercise(monkeypatch, repair="missing")
    assert not scorer._is_same_domain_unproven_web_identity(result.details["identity_receipt"])
    assert not scorer._has_explicitly_unproven_fit_dimensions(
        {"observed_company_name": "Acme", "observed_company_website": "https://acme.example",
         "observed_company_linkedin": ""},
        ("identity",), identity_receipt=result.details["identity_receipt"],
        investigation_receipt=result.details["investigation_receipt"],
    )


@pytest.mark.parametrize("repair", ["missing", "wrong_name"])
def test_missing_linkedin_terminal_zero_survives_public_retry_and_checkpoints(monkeypatch, repair):
    from lab_arena import scoring

    result, events, investigations = _exercise(monkeypatch, repair=repair)
    terminal = {
        "company_name": "Acme", "final_score": 0.0,
        "verifier_gate_receipts": [result.receipt("company_fit")],
    }
    attempts, checkpoints = [], []

    def judge(companies, _buyer, _reference):
        attempts.append([row["company_name"] for row in companies])
        rows = []
        for company in companies:
            if company["company_name"] == "Acme":
                rows.append(copy.deepcopy(terminal))
            else:
                rows.append({"company_name": "Other", "final_score": 0.0,
                             "failure_reason": "provider error"})
        return rows

    with pytest.raises(scoring.ScoringError):
        scoring.score_work_item(
            {"scored_run_id": "identity-finality-test"},
            icp={"employee_count": ["11-50"], "max_companies": 2},
            companies=[{"company_name": name, "employee_count": "11-50"}
                       for name in ("Acme", "Other")],
            scorer=judge, on_terminal_breakdowns=lambda rows: checkpoints.append(copy.deepcopy(rows)),
        )
    assert attempts == [["Acme", "Other"], ["Other"], ["Other"]]
    assert checkpoints == [[terminal]]
    assert events == ["lead_scorer_reverify", "lead_scorer_reverify_schema_repair", "investigate"]
    assert len(investigations) == 1


def test_changed_domain_keeps_existing_genuine_negative_without_identity_proof(monkeypatch):
    result, _events, _investigations = _exercise(monkeypatch, repair="wrong_domain")
    assert result.decision != "match"
    assert result.details["identity_decision"] != "match"
    assert not scorer._is_explicitly_missing_web_linkedin_identity(
        {"observed_company_name": "Acme", "observed_company_website": "https://other.example",
         "observed_company_linkedin": ""}, result.details["identity_receipt"],
    )


@pytest.mark.parametrize("field,value", [
    ("observed_company_name", None), ("observed_company_name", ""),
    ("observed_company_website", None), ("observed_company_website", ""),
    ("observed_company_website", "not a URL"),
    ("observed_company_linkedin", None), ("observed_company_linkedin", " "),
])
def test_missing_linkedin_proof_requires_complete_typed_identity_fields(field, value):
    verdict = _complete_verdict(observed_company_linkedin="")
    verdict[field] = value
    receipt = scorer._web_identity_receipt(_company(), verdict)
    assert not scorer._is_explicitly_missing_web_linkedin_identity(verdict, receipt)


@pytest.mark.parametrize("linkedin", ["", "not a LinkedIn URL"])
def test_missing_linkedin_proof_requires_valid_submitted_slug(linkedin):
    company = _company().model_copy(update={"company_linkedin": linkedin})
    verdict = _complete_verdict(observed_company_linkedin="")
    receipt = scorer._web_identity_receipt(company, verdict)
    assert not scorer._is_explicitly_missing_web_linkedin_identity(verdict, receipt)
