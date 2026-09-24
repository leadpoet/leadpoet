from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
)


def _prior_identity_result():
    return company_fit_match(
        "grounded identity",
        details={
            "identity_decision": COMPANY_FIT_MATCH,
            "identity_receipt": {
                "decision": COMPANY_FIT_MATCH,
                "reason_code": "verifier_accepted",
                "observed_name": "dbs",
                "observed_domain": "dbs.com",
                "observed_linkedin_slug": "dbs-bank",
            },
        },
    )


def _dbs_company() -> CompanyOutput:
    return CompanyOutput.model_construct(
        company_name="DBS",
        company_website="https://dbs.com",
        company_linkedin="",
        industry="Financial Services",
        sub_industry="Banking",
        employee_count="10,001+",
        company_stage="",
        country="Singapore",
        state="",
        description="",
        intent_details=None,
        fit_evidence_urls=[],
        company_stage_evidence=[],
        intent_signals=[],
        required_attribute=None,
    )


def test_repair_identity_scope_allows_same_canonical_tuple():
    repaired = {
        "decision": COMPANY_FIT_MATCH,
        "reason_code": "verifier_accepted",
        "observed_name": " DBS ",
        "observed_domain": "DBS.COM",
        "observed_linkedin_slug": "DBS-BANK",
    }

    assert not lead_scorer._schema_repair_changes_grounded_identity(
        _prior_identity_result(),
        repaired,
        repaired_dimensions=("required_attribute",),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_name", ""),
        ("observed_domain", ""),
        ("observed_linkedin_slug", ""),
        ("observed_linkedin_slug", None),
    ],
)
def test_repair_identity_scope_preserves_incomplete_prior_handling(field, value):
    prior = _prior_identity_result()
    prior.details["identity_receipt"][field] = value
    changed = {
        "decision": COMPANY_FIT_UNAVAILABLE,
        "reason_code": "identity_not_proven",
        "observed_name": "dbsbank",
        "observed_domain": "dbs.com",
        "observed_linkedin_slug": "dbs-bank",
    }

    assert not lead_scorer._schema_repair_changes_grounded_identity(
        prior,
        changed,
        repaired_dimensions=("required_attribute",),
    )


def test_repair_identity_scope_allows_only_fully_bound_same_entity_proof():
    company = _dbs_company()
    anchor = {
        "normalized_name": "dbs",
        "registrable_dns_domain": "dbs.com",
        "linkedin_company_slug": "dbs-bank",
    }
    repaired_verdict = {
        "observed_company_name": "DBS Bank Ltd",
        "observed_company_website": "https://www.dbs.com",
        "observed_company_linkedin": (
            "https://www.linkedin.com/company/dbs-bank"
        ),
    }
    arbitrary_claim = {
        "status": "VERIFIED",
        "evidence_url": "https://www.dbs.com/about",
        "evidence_quote": "DBS Bank Ltd",
    }
    unbound = lead_scorer._web_identity_receipt(
        company,
        repaired_verdict,
        verified_homepage_identity=anchor,
        verified_homepage_transport_domain="dbs.com",
        verified_rebrand_identity=arbitrary_claim,
    )
    assert lead_scorer._schema_repair_changes_grounded_identity(
        _prior_identity_result(),
        unbound,
        repaired_dimensions=("required_attribute",),
    )

    bound_proof = {
        "status": "VERIFIED",
        "old_name": "DBS",
        "new_name": "DBS Bank Ltd",
        "old_domain": "dbs.com",
        "new_domain": "dbs.com",
        "shared_linkedin_slug": "dbs-bank",
        "evidence_url": "https://www.dbs.com/about",
        "evidence_quote": "DBS Bank Ltd, trading as DBS",
    }
    bound = lead_scorer._web_identity_receipt(
        company,
        repaired_verdict,
        verified_homepage_identity=anchor,
        verified_homepage_transport_domain="dbs.com",
        verified_rebrand_identity=bound_proof,
    )
    assert bound["decision"] == COMPANY_FIT_MATCH
    assert bound["reason_code"] == "verified_same_domain_alias"
    assert not lead_scorer._schema_repair_changes_grounded_identity(
        _prior_identity_result(),
        bound,
        repaired_dimensions=("required_attribute",),
    )


def test_unrelated_repair_cannot_attach_old_stage_proof_to_new_dbs_entity(
    monkeypatch,
):
    initial = {
        "observed_company_name": "DBS",
        "observed_company_website": "https://www.dbs.com",
        "observed_company_linkedin": (
            "https://www.linkedin.com/company/dbs-bank"
        ),
        "observed_employee_count": "10,001+",
        "employee_size_matches": True,
        "employee_size_evidence_url": (
            "https://www.linkedin.com/company/dbs-bank"
        ),
        "employee_size_evidence_quote": "Company size 10,001+ employees",
        "observed_industry": "Financial Services",
        "observed_subindustry": "Banking",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://www.dbs.com/about",
        "industry_evidence_quote": "DBS provides banking services.",
        "observed_hq_country": "Singapore",
        "observed_hq_state": "",
        "geography_matches": True,
        "geography_evidence_url": "https://www.dbs.com/about",
        "geography_evidence_quote": "DBS is headquartered in Singapore.",
        "observed_company_stage": "",
        "stage_matches": None,
        "stage_evidence_url": "",
        "stage_evidence_quote": "",
        "attribute_satisfied": None,
        "attribute_evidence_url": "",
        "attribute_evidence_quote": "",
        "reason": "required attribute and stage need proof",
    }
    repaired = {
        **initial,
        "observed_company_name": "DBS Bank Ltd",
        "observed_company_stage": "Public",
        "stage_matches": True,
        "stage_evidence_url": "https://www.dbs.com/newsroom/listing",
        "stage_evidence_quote": (
            "DBS is headquartered and listed in Singapore."
        ),
        "attribute_satisfied": True,
        "attribute_evidence_url": "https://www.dbs.com/newsroom/product",
        "attribute_evidence_quote": "DBS Bank launched a payment product.",
        "reason": "repaired against DBS Bank Ltd",
    }
    calls = {"broad": 0, "investigator": 0, "ground": 0}

    async def provider(**_kwargs):
        calls["broad"] += 1
        return (initial if calls["broad"] == 1 else repaired), ""

    async def investigator(**kwargs):
        calls["investigator"] += 1
        assert kwargs["targets"] == ("stage",)
        finding = {
            "target": "stage",
            "status": "VERIFIED",
            "observed_value": "Public",
            "observed_country": "",
            "observed_state": "",
            "observed_industry": "",
            "observed_subindustry": "",
            "activity_role": "unresolved",
            "evidence_url": "https://www.dbs.com/investors/listing",
            "evidence_quote": (
                "DBS ordinary shares are listed on the Singapore Exchange "
                "under ticker D05."
            ),
            "old_name": "",
            "new_name": "",
            "old_domain": "",
            "new_domain": "",
            "shared_linkedin_slug": "",
            "reason": "verified",
        }
        return {
            "claims": {"stage": finding},
            "_validated_stage_finding": finding,
            "failure_reason": "",
            "usage": {
                "reasoning_turns": 1,
                "search_calls": 0,
                "fetch_calls": 0,
            },
        }

    original_ground = lead_scorer._ground_required_attribute_evidence

    async def counted_ground(*args, **kwargs):
        calls["ground"] += 1
        return await original_ground(*args, **kwargs)

    async def keep_employee_observation(candidate, *_args, **_kwargs):
        return candidate

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigator)
    monkeypatch.setattr(
        lead_scorer,
        "_ground_required_attribute_evidence",
        counted_ground,
    )
    monkeypatch.setattr(
        lead_scorer,
        "_refresh_linkedin_employee_size_observation",
        keep_employee_observation,
    )
    icp = SimpleNamespace(
        required_attribute="Operates banking services and launched a payment product.",
        company_stage="Public",
        employee_count="10,001+",
        industry="Financial Services",
        sub_industry="Banking",
        product_service="Banking services",
        geography="Singapore",
        country="Singapore",
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _dbs_company(),
            icp,
            require_company_fit_dimensions=True,
            evidence_investigator=True,
        )
    )

    assert calls == {"broad": 2, "investigator": 1, "ground": 1}
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["identity_decision"] == COMPANY_FIT_MATCH
    assert result.details["identity_receipt"]["observed_name"] == "dbs"
    assert result.details["required_attribute_decision"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["dimension_evidence"]["stage"] == {
        "url": "https://www.dbs.com/investors/listing",
        "quote": (
            "DBS ordinary shares are listed on the Singapore Exchange "
            "under ticker D05."
        ),
    }
    assert result.details["dimension_evidence"]["required_attribute"] == {
        "url": "",
        "quote": "",
    }
    assert result.details["provider_observations"]["observed_company_name"] == (
        "DBS"
    )
