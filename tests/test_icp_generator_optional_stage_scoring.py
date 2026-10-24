"""New optional stages pass through the existing Arena scoring boundary."""

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from gateway.tasks import icp_generator as generator
from qualification.scoring import lead_scorer as scorer
from qualification.scoring.company_fit_decision import company_fit_match
from qualification.scoring.competition import _normalized_icp


@pytest.mark.parametrize(
    ("staged", "stage_decision", "stage_proof", "expected"),
    [
        (False, "unavailable", False, "match"),
        (False, "mismatch", False, "match"),
        (True, "match", True, "match"),
        (True, "mismatch", True, "mismatch"),
        (True, "unavailable", False, "unavailable"),
        (True, "match", False, "unavailable"),
    ],
)
def test_generated_stage_filters_use_existing_company_fit_contract(
    monkeypatch, staged, stage_decision, stage_proof, expected,
):
    monkeypatch.setenv("LAB_ARENA_ICP_EXCLUSIONS_ENABLED", "0")
    rows, _, _ = generator.generate_icp_set(20261025, 10, base_seed=42)
    row = next(row for row in rows if bool(row["company_stage"]) is staged)
    icp = ICPPrompt.model_validate(_normalized_icp(row))
    assert ICPPrompt.model_validate_json(icp.model_dump_json()) == icp
    company = CompanyOutput(
        company_name="Acme",
        company_website="https://acme.com",
        company_linkedin="https://linkedin.com/company/acme",
        industry=row["industry"],
        sub_industry=row["sub_industry"],
        employee_count=row["employee_count"][0],
        company_stage=row["company_stage"] or "Public",
        country=row["country"],
        intent_signals=[{
            "description": "Announced a new workflow product.",
            "source": "news", "url": "https://acme.com/news/product",
            "date": "2026-10-01", "snippet": "Announced a new workflow product.",
            "matched_icp_signal": 0,
        }],
    )

    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def identity(*_args, **_kwargs):
        return company_fit_match("identity verified")

    async def web(*_args, **kwargs):
        assert kwargs["require_company_fit_dimensions"] is True
        evidence = {
            dimension: {"url": f"https://acme.com/{dimension}", "quote": f"Verified {dimension}"}
            for dimension in ("employee_size", "industry", "geography", "required_attribute")
        }
        if stage_proof:
            evidence["stage"] = {"url": "https://acme.com/stage", "quote": "Verified funding stage"}
        return company_fit_match(
            "web observation",
            details={
                "dimension_decisions": {
                    "employee_size": "match", "industry": "match", "geography": "match", "stage": stage_decision,
                },
                "required_attribute_decision": "match",
                "identity_decision": "match",
                "identity_receipt": {
                    "decision": "match", "reason_code": "verifier_accepted",
                    "submitted_name": "acme", "submitted_domain": "acme.com", "submitted_linkedin_slug": "acme",
                    "observed_name": "acme", "observed_domain": "acme.com", "observed_linkedin_slug": "acme",
                    "evidence_source": "company_web_reverification",
                },
                "dimension_evidence": evidence,
            },
        )

    monkeypatch.setattr(scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(scorer, "verify_company_exists", identity)
    monkeypatch.setattr(scorer, "_llm_reverify_company", web)
    result = asyncio.run(scorer._verify_company_fit(
        company, icp, 0.0, 1.0, set(), require_https_transport=True,
    ))
    assert result.decision == expected
    assert result.details["company_fit_stage_required"] is staged
    dimensions = result.details["company_fit_dimensions"]
    assert {key: dimensions[key] for key in ("identity", "employee_size", "industry", "geography")} == {
        "identity": "match", "employee_size": "match", "industry": "match", "geography": "match",
    }
    assert result.details["required_attribute_decision"] == "match"
