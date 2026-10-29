"""Completed industry research must preserve independent unfinished work."""

import asyncio
import copy

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from lab_arena import scoring
from qualification.scoring import competition, lead_scorer
from qualification.scoring.company_fit_decision import COMPANY_FIT_MISMATCH


def _company():
    return CompanyOutput(
        company_name="Acme", company_website="https://acme.example",
        company_linkedin="https://www.linkedin.com/company/acme",
        industry="Software", employee_count="51-200", country="United States",
        intent_signals=[{
            "description": "Raised a round", "source": "news",
            "url": "https://acme.example/news", "date": "2026-09-01",
            "snippet": "Acme raised a round.",
        }],
    )


def _verdict(mismatch=False):
    return {
        "observed_company_name": "Acme",
        "observed_company_website": "https://acme.example",
        "observed_company_linkedin": "https://www.linkedin.com/company/acme",
        "observed_employee_count": None, "employee_size_matches": None,
        "employee_size_evidence_url": "", "employee_size_evidence_quote": "",
        "observed_industry": "Software", "observed_subindustry": "SaaS",
        "industry_matches": True, "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://acme.example/about",
        "industry_evidence_quote": "Acme supplies SaaS software.",
        "observed_hq_country": "Canada" if mismatch else "",
        "observed_hq_state": "", "geography_matches": False if mismatch else None,
        "geography_evidence_url": "https://acme.example/about" if mismatch else "",
        "geography_evidence_quote": "Acme is headquartered in Canada." if mismatch else "",
        "observed_company_stage": "", "stage_matches": None,
        "stage_evidence_url": "", "stage_evidence_quote": "",
        "reason": "bounded company research",
    }


@pytest.mark.parametrize("phase", ["initial", "post_repair"])
@pytest.mark.parametrize("fault", [
    None, "profile", "profile_provider", "provider", "malformed", "source",
    "incomplete", "identity",
])
def test_semantic_review_preserves_unfinished_obligations_through_arena(
    monkeypatch, phase, fault,
):
    _check_case(monkeypatch, phase, fault)


@pytest.mark.parametrize("phase", ["initial", "post_repair"])
def test_independently_proved_mismatch_remains_a_valid_negative(monkeypatch, phase):
    _check_case(monkeypatch, phase, "profile", mismatch=True)


@pytest.mark.parametrize("phase", ["initial", "post_repair"])
def test_completed_profile_failure_matches_later_settlement(monkeypatch, phase):
    """The semantic exit uses the same bounded local exhaustion as the later path."""
    outcomes = []
    for semantic_enabled in (False, True):
        with monkeypatch.context() as context:
            outcomes.append(_check_case(
                context, phase, "profile", semantic_enabled=semantic_enabled,
            ))
    assert outcomes[0] == outcomes[1] == {
        "failure_class": "employee_size_verification_failed",
        "failure_reason_code": "source_blocked",
        "attempts": 3,
        "terminal_class": "company_verification_exhausted",
    }


def _check_case(monkeypatch, phase, fault, mismatch=False, semantic_enabled=True):
    company = _company()
    icp = ICPPrompt(
        icp_id="target", prompt="target", industry="Software", sub_industry="SaaS",
        employee_count="51-200", company_stage="Series A", geography="United States",
        country="United States", product_service="software",
    )
    requests = []
    target_calls = []
    semantic_calls = []
    investigations = []

    async def request(**kwargs):
        requests.append(kwargs["telemetry_purpose"])
        verdict = _verdict(mismatch and (phase == "initial" or len(requests) > 1))
        if fault == "identity":
            for field in ("observed_company_name", "observed_company_website", "observed_company_linkedin"):
                verdict[field] = ""
        return verdict, ""

    async def refresh(verdict, _company, _icp, *, invocation_cache, **_kwargs):
        if fault in {"profile", "profile_provider"}:
            reason = "provider_error" if fault == "profile_provider" else "source_blocked"
            invocation_cache.update(
                refresh_outcome="retryable_failure", structured_failure=True,
                structured_failure_reason=reason,
                evidence={"outcome": "insufficient_evidence"},
            )
            invocation_cache[lead_scorer.VERIFIER_FAILURE_DETAIL_KEY] = reason
        else:
            invocation_cache["refresh_outcome"] = "verified"
        return dict(verdict)

    def targets(*_args, **_kwargs):
        target_calls.append(True)
        if phase == "post_repair" and len(target_calls) == 1:
            return ()
        return ("headcount", "industry", "geography", "stage")

    def semantic(*_args):
        semantic_calls.append(True)
        return semantic_enabled and (phase == "initial" or len(semantic_calls) > 1)

    async def investigate(**kwargs):
        investigations.append(kwargs)
        failure = {"provider": "provider_error", "malformed": "malformed_response"}.get(fault, "")
        findings = {target: {
            "target": target, "status": "UNPROVEN", "observed_value": None,
            "observed_country": "", "observed_state": "", "observed_industry": "",
            "observed_subindustry": "", "activity_role": "unresolved",
            "evidence_url": "", "evidence_quote": "", "reason": "no proof",
        } for target in kwargs["targets"]}
        outcomes = [{"ok": True, "url_sha256": "a" * 64, "loaded_text_length": 100}]
        if fault == "source":
            outcomes.append({"ok": False, "url_sha256": "b" * 64, "error_class": "http_error"})
        return {
            "claims": findings, "_completed_submit": fault not in {"provider", "malformed", "incomplete"},
            "failure_reason": failure or ("admission_budget_interrupted" if fault == "incomplete" else ""),
            "usage": {"fetch_outcomes": outcomes},
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(lead_scorer, "_refresh_linkedin_employee_size_observation", refresh)
    monkeypatch.setattr(lead_scorer, "_targeted_company_investigation_dimensions", targets)
    monkeypatch.setattr(lead_scorer, "_positive_semantic_review_needed", semantic)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    result = asyncio.run(lead_scorer._llm_reverify_company(
        company, icp, require_company_fit_dimensions=True, evidence_investigator=True,
    ))
    assert len(investigations) == 1
    assert investigations[0]["positive_semantic_review"] is semantic_enabled
    assert len(requests) == (1 if phase == "initial" and fault != "identity" else 2)
    if mismatch:
        assert result.decision == COMPANY_FIT_MISMATCH
    else:
        assert result.decision == "unavailable"
        assert (result.details.get("failure_class") == "insufficient_fit_evidence") is (fault is None)

    # Use the real fit receipt producer and Arena terminal classifiers.
    # A completed source-local failure retains the frozen bounded retry rule;
    # provider faults and unfinished research remain systemic and retryable.
    fit = lead_scorer._complete_company_fit_result(
        result.decision, result.reason,
        dimensions=result.details["dimension_decisions"],
        dimension_evidence=result.details["dimension_evidence"], stage_required=True,
        supporting_receipts=[result.details["investigation_receipt"]],
        failure_class=result.details.get("failure_class", ""),
        failure_reason_code=result.details.get(lead_scorer.VERIFIER_FAILURE_DETAIL_KEY, ""),
    )
    row = {
        "final_score": 0.0, "failure_reason": "Company fit unavailable" if fit.decision == "unavailable" else "",
        "verifier_gate_receipts": [fit.receipt("company_fit")],
    }
    retryable = fault is not None and not mismatch
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure(row) is retryable
    local_exhaustion = fault == "profile" and not mismatch
    if local_exhaustion:
        assert result.details["failure_class"] == "employee_size_verification_failed"
        assert result.details[lead_scorer.VERIFIER_FAILURE_DETAIL_KEY] == "source_blocked"
        assert competition.scorer_breakdown_has_company_local_verification_failure(row)
    elif retryable:
        assert not competition.scorer_breakdown_has_company_local_verification_failure(row)
        with pytest.raises(competition.CompetitionScorerInputError):
            competition.terminal_company_verification_breakdown(row)
    calls = []

    def scorer(*_args):
        calls.append(True)
        return [copy.deepcopy(row)]

    run = lambda: scoring.score_work_item(
        {"scored_run_id": "test"}, icp={"employee_count": ["51-200"], "max_companies": 1},
        companies=[{"company_name": "Acme", "employee_count": "51-200"}], scorer=scorer,
    )
    terminal_class = None
    if local_exhaustion:
        terminal = run()[0]
        assert len(calls) == 3
        assert terminal["final_score"] == 0.0
        terminal_class = terminal["verifier_gate_receipts"][0]["failure_class"]
        assert terminal_class == "company_verification_exhausted"
        assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(terminal)
    elif retryable:
        with pytest.raises(scoring.ScoringError):
            run()
        assert len(calls) == 3
    else:
        assert run()[0]["final_score"] == 0.0
        assert len(calls) == 1
    return {
        "failure_class": result.details.get("failure_class"),
        "failure_reason_code": result.details.get(lead_scorer.VERIFIER_FAILURE_DETAIL_KEY),
        "attempts": len(calls), "terminal_class": terminal_class,
    }
