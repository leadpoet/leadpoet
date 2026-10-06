from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer, linkedin_company_size
from qualification.scoring.competition import _normalized_company, _normalized_icp
from qualification.scoring.company_fit_decision import company_fit_match


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("none", "match"),
        ("outside_requested_range", "mismatch"),
        ("wrong_slug", "unavailable"),
        ("wrong_domain", "unavailable"),
        ("invalid_range", "unavailable"),
        ("no_range", "unavailable"),
        ("bad_status", "unavailable"),
        ("no_company_type", "match"),
    ],
)
def test_public_stage_fetch_retains_exact_bound_size(monkeypatch, mutation, expected):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/oct06_marqeta_bound_profile_size.json")
        .read_text()
    )
    payload = fixture["structured_profile_response"]
    element = payload["result"]["data"]["element"]
    if mutation == "wrong_slug":
        element["linkedinUrl"] = "https://www.linkedin.com/company/unrelated"
    elif mutation == "wrong_domain":
        element["website"] = "https://unrelated.example"
    elif mutation == "invalid_range":
        element["employeeCountRange"] = {"start": 500, "end": 1000}
    elif mutation == "no_range":
        element.pop("employeeCountRange")
    elif mutation == "outside_requested_range":
        element["employeeCountRange"] = {"start": 11, "end": 50}
    elif mutation == "no_company_type":
        element.pop("companyType")
    elif mutation == "bad_status":
        payload["result"]["data"]["status"] = 404
    fetches = []
    investigated = []

    async def provider(**_kwargs):
        return copy.deepcopy(fixture["verifier_response"]), ""

    async def structured_fetch(domain, url, *, public_company_evidence, **_kwargs):
        fetches.append((domain, url))
        company_type = linkedin_company_size.project_structured_linkedin_company_type(
            domain, url, payload
        )
        if company_type:
            public_company_evidence.update(company_type)
        return linkedin_company_size.project_structured_linkedin_company_size(
            domain, url, payload
        )

    async def unexpected_profile_fetch(*_args, **_kwargs):
        pytest.fail("The mismatched model profile must not trigger a size fetch")

    async def ground_attribute(verdict, **_kwargs):
        # Attribute grounding is a separate gate, outside this size regression.
        return verdict, {}

    async def investigate(**kwargs):
        investigated.append(kwargs["investigation_targets"])
        return kwargs["verdict"], kwargs["prior_result"], {}, {}, {}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("DEEPLINE_API_KEY", raising=False)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "_ground_required_attribute_evidence", ground_attribute)
    monkeypatch.setattr(
        lead_scorer, "fetch_structured_linkedin_company_size", structured_fetch
    )
    monkeypatch.setattr(
        lead_scorer, "fetch_current_linkedin_company_size", unexpected_profile_fetch
    )
    monkeypatch.setattr(
        lead_scorer, "_run_targeted_company_evidence_investigation", investigate
    )
    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            CompanyOutput.model_validate(_normalized_company(
                fixture["submitted_company"], integrity_policy=True, company_quality=True
            )),
            ICPPrompt.model_validate(_normalized_icp(fixture["icp"])),
            require_company_fit_dimensions=True,
            company_quality=True,
            evidence_investigator=True,
            verified_homepage_identity=company_fit_match(
                "independently verified homepage identity",
                details={
                    "actual_final_url": "https://marqeta.com/",
                    "identity": {
                        "decision": "match",
                        "evidence_source": "company_homepage",
                        "observed_name": "marqeta",
                        "observed_domain": "marqeta.com",
                        "observed_linkedin_slug": "marqeta",
                    },
                },
            ),
        )
    )
    assert fetches == [("marqeta.com", "https://www.linkedin.com/company/marqeta")]
    assert result.details["dimension_decisions"]["employee_size"] == expected
    assert investigated
    assert ("headcount" in investigated[0]) == (expected != "match")
    # A stored Public Company label remains insufficient for current stage.
    assert result.details["dimension_decisions"]["stage"] == "unavailable"
    if expected != "unavailable":
        evidence = result.details["dimension_evidence"]["employee_size"]
        assert evidence["source_field"] == "employeeCountRange"
        assert evidence["url"] == "https://www.linkedin.com/company/marqeta"
        assert evidence["employee_count"] == (
            "11-50" if mutation == "outside_requested_range" else "501-1,000"
        )
