from __future__ import annotations

import asyncio
from datetime import date
import json
from types import SimpleNamespace

from gateway.qualification.models import CompanyOutput, ICPPrompt
from lab_arena import scoring as arena_scoring
from lab_arena import operations as arena_operations
from qualification.scoring.competition import CompetitionCompanyScorer
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.company_evidence_investigator import (
    _validated_findings,
)
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
)
from qualification.scoring.linkedin_company_size import (
    MALFORMED_RESPONSE_FAILURE_REASON,
    PROVIDER_ERROR_FAILURE_REASON,
    VERIFIER_FAILURE_REASON_KEY,
)
from qualification.scoring.lead_scorer import (
    _employee_size_sources_conflict,
    _project_investigator_headcount,
    _refresh_linkedin_employee_size_observation,
    _reverify_decision,
    _stage_quote_supports_observation,
    _targeted_company_investigation_dimensions,
)


def _company(
    *,
    name: str = "Acme",
    website: str = "https://acme.example",
    linkedin: str = "https://www.linkedin.com/company/acme",
) -> CompanyOutput:
    return CompanyOutput(
        company_name=name,
        company_website=website,
        company_linkedin=linkedin,
        industry="Software",
        employee_count="11-50",
        country="United States",
        state="California",
        intent_signals=[{
            "description": "raised",
            "source": "news",
            "url": "https://news.example/acme",
            "date": "2026-09-01",
            "snippet": "Acme raised a round.",
        }],
    )


def _icp(**overrides) -> ICPPrompt:
    values = {
        "icp_id": "target",
        "prompt": "target",
        "industry": "Software",
        "sub_industry": "SaaS",
        "employee_count": "11-50",
        "company_stage": "",
        "geography": "United States",
        "product_service": "software",
    }
    values.update(overrides)
    return ICPPrompt(**values)


def _complete_verdict(**overrides):
    verdict = {
        "observed_company_name": "Acme",
        "observed_company_website": "https://acme.example",
        "observed_company_linkedin": "https://www.linkedin.com/company/acme",
        "observed_employee_count": "11-50",
        "employee_size_matches": True,
        "employee_size_evidence_url": "https://evidence.example/headcount",
        "employee_size_evidence_quote": "Acme has 11-50 employees.",
        "observed_industry": "Software",
        "observed_subindustry": "SaaS",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://evidence.example/industry",
        "industry_evidence_quote": "Acme supplies SaaS software.",
        "observed_hq_country": "United States",
        "observed_hq_state": "California",
        "geography_matches": True,
        "geography_evidence_url": "https://evidence.example/hq",
        "geography_evidence_quote": "Acme is headquartered in California, United States.",
        "reason": "verified",
    }
    verdict.update(overrides)
    return verdict


def _finding(target: str, **overrides):
    finding = {
        "target": target,
        "status": "VERIFIED",
        "observed_value": "Public",
        "evidence_url": "https://acme.example/investors",
        "evidence_quote": "Acme common stock is listed on NASDAQ under ticker ACME.",
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "verified",
    }
    finding.update(overrides)
    return finding


def test_decisive_quote_must_occur_in_fetched_page():
    url = "https://acme.example/investors"
    finding = _finding("stage", evidence_url=url)
    accepted = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={url: "Acme common stock is listed on NASDAQ under ticker ACME."},
        first_party_domains={"acme.example"},
    )
    assert accepted["stage"]["status"] == "VERIFIED"

    rejected = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={url: "Acme is a privately held software company."},
        first_party_domains={"acme.example"},
    )
    assert rejected["stage"]["status"] == "UNPROVEN"
    assert rejected["stage"]["evidence_url"] == ""

    wrong_company = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={url: "Acme common stock is listed on NASDAQ under ticker ACME."},
        first_party_domains={"acme.example"},
        identity_names={"different company"},
    )
    assert wrong_company["stage"]["status"] == "UNPROVEN"

    locator_snippet_only = _validated_findings(
        {"findings": [finding]},
        targets=("stage",),
        fetched_pages={},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert locator_snippet_only["stage"]["status"] == "UNPROVEN"


def test_rebrand_needs_first_party_explicit_old_and_new_name_continuity():
    url = "https://help.wayground.com/rebrand"
    explicit = _finding(
        "rebrand",
        observed_value="Wayground",
        evidence_url=url,
        evidence_quote="Quizizz is now Wayground following our rebrand.",
        old_name="Quizizz",
        new_name="Wayground",
        old_domain="quizizz.com",
        new_domain="wayground.com",
        shared_linkedin_slug="quizizz",
    )
    accepted = _validated_findings(
        {"findings": [explicit]},
        targets=("rebrand",),
        fetched_pages={
            url: (
                "Quizizz is now Wayground following our rebrand. "
                "The domain changed from quizizz.com to wayground.com."
            )
        },
        first_party_domains={"quizizz.com", "wayground.com"},
    )
    assert accepted["rebrand"]["status"] == "VERIFIED"

    redirect_only = dict(explicit)
    redirect_only["evidence_quote"] = "Visit our new website at Wayground."
    rejected = _validated_findings(
        {"findings": [redirect_only]},
        targets=("rebrand",),
        fetched_pages={url: "Visit our new website at Wayground."},
        first_party_domains={"quizizz.com", "wayground.com"},
    )
    assert rejected["rebrand"]["status"] == "UNPROVEN"

    prospective = dict(explicit)
    prospective["evidence_quote"] = "Quizizz plans to rebrand as Wayground."
    prospective_result = _validated_findings(
        {"findings": [prospective]},
        targets=("rebrand",),
        fetched_pages={
            url: (
                "Quizizz plans to rebrand as Wayground. "
                "quizizz.com wayground.com"
            )
        },
        first_party_domains={"quizizz.com", "wayground.com"},
    )
    assert prospective_result["rebrand"]["status"] == "UNPROVEN"


def test_verified_rebrand_is_a_separate_identity_proof_not_a_domain_rewrite():
    company = _company(
        name="Wayground formerly Quizizz",
        website="https://quizizz.com",
        linkedin="https://www.linkedin.com/company/quizizz",
    )
    verdict = _complete_verdict(
        observed_company_name="Wayground",
        observed_company_website="https://wayground.com",
        observed_company_linkedin="https://www.linkedin.com/company/quizizz",
    )
    assert _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        company_quality=True,
    ).decision == COMPANY_FIT_MISMATCH

    proof = _finding(
        "rebrand",
        observed_value="Wayground",
        evidence_url="https://help.wayground.com/rebrand",
        evidence_quote="Quizizz is now Wayground following our rebrand.",
        old_name="Quizizz",
        new_name="Wayground",
        old_domain="quizizz.com",
        new_domain="wayground.com",
        shared_linkedin_slug="quizizz",
    )
    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity=proof,
        company_quality=True,
    )
    assert result.decision == COMPANY_FIT_MATCH
    receipt = result.details["identity_receipt"]
    assert receipt["submitted_domain"] == "quizizz.com"
    assert receipt["observed_domain"] == "wayground.com"
    assert receipt["reason_code"] == "verified_rebrand_continuity"


def test_rebrand_proof_cannot_bind_an_unrelated_linkedin_company():
    company = _company(
        name="Wayground formerly Quizizz",
        website="https://quizizz.com",
        linkedin="https://www.linkedin.com/company/quizizz",
    )
    verdict = _complete_verdict(
        observed_company_name="Wayground",
        observed_company_website="https://wayground.com",
        observed_company_linkedin="https://www.linkedin.com/company/unrelated",
    )
    proof = _finding(
        "rebrand",
        observed_value="Wayground",
        evidence_url="https://help.wayground.com/rebrand",
        evidence_quote="Quizizz is now Wayground following our rebrand.",
        old_name="Quizizz",
        new_name="Wayground",
        old_domain="quizizz.com",
        new_domain="wayground.com",
        shared_linkedin_slug="quizizz",
    )
    assert _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity=proof,
        company_quality=True,
    ).decision == COMPANY_FIT_MISMATCH


def test_rebrand_proof_binds_composite_observed_name_symmetrically():
    company = _company(
        name="Quizizz",
        website="https://quizizz.com",
        linkedin="https://www.linkedin.com/company/quizizz",
    )
    verdict = _complete_verdict(
        observed_company_name="Wayground (formerly Quizizz)",
        observed_company_website="https://wayground.com",
        observed_company_linkedin="https://www.linkedin.com/company/quizizz",
    )
    proof = _finding(
        "rebrand",
        observed_value="Wayground",
        evidence_url="https://help.wayground.com/rebrand",
        evidence_quote="Quizizz is now Wayground following our rebrand.",
        old_name="Quizizz",
        new_name="Wayground",
        old_domain="quizizz.com",
        new_domain="wayground.com",
        shared_linkedin_slug="quizizz",
    )

    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity=proof,
        company_quality=True,
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["identity_receipt"]["reason_code"] == (
        "verified_rebrand_continuity"
    )


def test_conflicting_current_headcount_is_unproven():
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_quote="PitchBook lists 7 employees.",
    )
    structured = {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example",
    }
    assert _employee_size_sources_conflict(verdict, structured) is True
    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=_company(),
        structured_employee_size_evidence=structured,
        employee_size_conflict=True,
        company_quality=True,
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["employee_size"] == COMPANY_FIT_UNAVAILABLE
    assert result.details["employee_size_conflict_receipt"] == {
        "status": "UNPROVEN",
        "reason_code": "conflicting_current_headcount",
        "resolution": "unresolved",
        "web_evidence": {
            "url": "https://evidence.example/headcount",
            "quote": "PitchBook lists 7 employees.",
        },
        "structured_evidence": structured,
    }


def test_exact_entity_linkedin_range_is_primary_over_third_party_exact_estimate():
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_url="https://pitchbook.example/acme",
        employee_size_evidence_quote="Acme has 7 employees.",
    )
    structured = {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example/",
    }
    anchor = {
        "normalized_name": "Acme",
        "registrable_dns_domain": "acme.example",
        "linkedin_company_slug": "acme",
    }

    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=_company(),
        verified_homepage_identity=anchor,
        structured_employee_size_evidence=structured,
        employee_size_conflict=True,
        company_quality=True,
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["employee_size"] == COMPANY_FIT_MATCH
    receipt = result.details["employee_size_conflict_receipt"]
    assert receipt["status"] == "VERIFIED"
    assert receipt["resolution"] == (
        "structured_linkedin_employee_count_range_primary"
    )
    assert receipt["primary_method"] == (
        "harvestapi_exact_company_employeeCountRange"
    )
    assert receipt["evaluation_date"]
    assert receipt["primary_source_url"] == (
        "https://www.linkedin.com/company/acme"
    )
    assert _targeted_company_investigation_dimensions(
        result,
        icp_stage="",
        employee_size_conflict=True,
    ) == ()

    disjoint = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(employee_count="51-200"),
        company=_company(),
        verified_homepage_identity=anchor,
        structured_employee_size_evidence=structured,
        employee_size_conflict=True,
        company_quality=True,
    )
    assert disjoint.decision == COMPANY_FIT_MISMATCH
    assert disjoint.details["employee_size_conflict_receipt"]["status"] == (
        "CONTRADICTED"
    )
    assert _targeted_company_investigation_dimensions(
        disjoint,
        icp_stage="",
        employee_size_conflict=True,
    ) == ()


def test_linkedin_primary_rejects_wrong_identity_member_count_and_linkedin_conflict():
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_url="https://pitchbook.example/acme",
        employee_size_evidence_quote="Acme has 7 employees.",
    )
    structured = {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example/",
    }
    anchor = {
        "normalized_name": "Acme",
        "registrable_dns_domain": "acme.example",
        "linkedin_company_slug": "acme",
    }

    for rejected_evidence, rejected_verdict in (
        (
            dict(structured, url="https://www.linkedin.com/company/unrelated"),
            verdict,
        ),
        (
            dict(structured, source_field="employeeCount"),
            verdict,
        ),
        (
            structured,
            dict(
                verdict,
                employee_size_evidence_url=(
                    "https://www.linkedin.com/company/acme"
                ),
            ),
        ),
    ):
        result = _reverify_decision(
            rejected_verdict,
            "",
            "",
            icp=_icp(),
            company=_company(),
            verified_homepage_identity=anchor,
            structured_employee_size_evidence=rejected_evidence,
            employee_size_conflict=True,
            company_quality=True,
        )
        assert result.decision == COMPANY_FIT_UNAVAILABLE
        assert result.details["employee_size_conflict_receipt"]["resolution"] == (
            "unresolved"
        )


def test_arena_conflict_check_collects_structured_size_without_replacing_direct_proof(
    monkeypatch,
):
    structured = {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example",
    }
    calls = []

    async def fake_fetch(domain, profile_url, *, diagnostic):
        del diagnostic
        calls.append((domain, profile_url))
        return structured

    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        fake_fetch,
    )
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_url="https://pitchbook.example/acme",
        employee_size_evidence_quote="Acme has 7 employees.",
    )
    cache = {}
    refreshed = asyncio.run(_refresh_linkedin_employee_size_observation(
        verdict,
        _company(),
        _icp(),
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        invocation_cache=cache,
        collect_structured_conflict=True,
    ))

    assert refreshed == verdict
    assert cache["structured_evidence"] == structured
    assert calls == [
        ("acme.example", "https://www.linkedin.com/company/acme")
    ]
    assert _employee_size_sources_conflict(refreshed, structured) is True


def test_structured_conflict_check_does_not_expand_calls_for_matching_headcount(
    monkeypatch,
):
    async def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("matching direct evidence must not add a provider call")

    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        unexpected_fetch,
    )
    verdict = _complete_verdict()
    cache = {}

    refreshed = asyncio.run(_refresh_linkedin_employee_size_observation(
        verdict,
        _company(),
        _icp(),
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        invocation_cache=cache,
        collect_structured_conflict=True,
    ))

    assert refreshed == verdict
    assert "structured_attempted" not in cache


def test_non_arena_mismatch_does_not_add_structured_fetch(monkeypatch):
    async def unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("ordinary scoring must not add a structured provider call")

    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        unexpected_fetch,
    )
    verdict = _complete_verdict(
        observed_employee_count=7,
        employee_size_matches=False,
        employee_size_evidence_url="https://pitchbook.example/acme",
        employee_size_evidence_quote="Acme has 7 employees.",
    )
    cache = {}
    refreshed = asyncio.run(_refresh_linkedin_employee_size_observation(
        verdict,
        _company(),
        _icp(),
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        invocation_cache=cache,
        collect_structured_conflict=False,
    ))

    assert refreshed == verdict
    assert "structured_attempted" not in cache


def test_fetched_current_headcount_repairs_a_nonconflicting_false_negative():
    verdict = _complete_verdict(
        observed_employee_count="2-10",
        employee_size_matches=False,
        employee_size_evidence_quote="Acme has 2-10 employees.",
    )
    finding = _finding(
        "headcount",
        observed_value=27,
        evidence_url="https://acme.example/about",
        evidence_quote="Acme has 27 employees company-wide.",
    )
    projected = _project_investigator_headcount(
        verdict,
        finding,
        icp=_icp(),
        existing_conflict=False,
    )
    assert projected["observed_employee_count"] == 27
    assert projected["employee_size_matches"] is True

    unchanged = _project_investigator_headcount(
        verdict,
        finding,
        icp=_icp(),
        existing_conflict=True,
    )
    assert unchanged["observed_employee_count"] == "2-10"

    canonical_band = dict(finding, observed_value="11-50")
    projected_band = _project_investigator_headcount(
        verdict,
        canonical_band,
        icp=_icp(),
        existing_conflict=False,
    )
    assert projected_band["observed_employee_count"] == "11-50"
    assert projected_band["employee_size_matches"] is True

    for invalid_value in ("11ish-50ish", "100000-200000"):
        rejected = _project_investigator_headcount(
            verdict,
            dict(finding, observed_value=invalid_value),
            icp=_icp(),
            existing_conflict=False,
        )
        assert rejected["observed_employee_count"] == "2-10"


def test_conflicting_third_party_ranges_remain_unproven():
    verdict = _complete_verdict(
        observed_employee_count="11-50",
        employee_size_matches=True,
        employee_size_evidence_url="https://directory.example/acme",
        employee_size_evidence_quote="Acme Company size 11-50 employees.",
    )
    structured = {
        "employee_count": "51-200",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example/",
    }
    assert _employee_size_sources_conflict(verdict, structured) is True

    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=_company(),
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        structured_employee_size_evidence=structured,
        employee_size_conflict=True,
        company_quality=True,
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    receipt = result.details["employee_size_conflict_receipt"]
    assert receipt["status"] == "UNPROVEN"
    assert receipt["resolution"] == "unresolved"


def test_headcount_finding_binds_value_and_rejects_scoped_counts():
    url = "https://acme.example/about"
    finding = _finding(
        "headcount",
        observed_value=27,
        evidence_url=url,
        evidence_quote="Acme has 27 employees company-wide.",
    )
    accepted = _validated_findings(
        {"findings": [finding]},
        targets=("headcount",),
        fetched_pages={url: finding["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert accepted["headcount"]["status"] == "VERIFIED"

    linkedin_band = dict(
        finding,
        observed_value="11-50",
        evidence_quote="Acme Company size 11-50 employees.",
    )
    accepted_band = _validated_findings(
        {"findings": [linkedin_band]},
        targets=("headcount",),
        fetched_pages={url: linkedin_band["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert accepted_band["headcount"]["status"] == "VERIFIED"

    mismatched_value = dict(finding, observed_value=99)
    rejected_value = _validated_findings(
        {"findings": [mismatched_value]},
        targets=("headcount",),
        fetched_pages={url: finding["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert rejected_value["headcount"]["status"] == "UNPROVEN"

    office_count = dict(
        finding,
        evidence_quote="Acme's London office has 27 employees.",
    )
    rejected_scope = _validated_findings(
        {"findings": [office_count]},
        targets=("headcount",),
        fetched_pages={url: office_count["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert rejected_scope["headcount"]["status"] == "UNPROVEN"

    associated_members = dict(
        finding,
        evidence_quote="Acme has 27 associated members on LinkedIn.",
    )
    rejected_members = _validated_findings(
        {"findings": [associated_members]},
        targets=("headcount",),
        fetched_pages={url: associated_members["evidence_quote"]},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert rejected_members["headcount"]["status"] == "UNPROVEN"

    for non_headcount_quote in (
        "Acme was founded in 2020.",
        "Acme reported 27 million dollars in annual revenue.",
    ):
        unrelated_number = dict(
            finding,
            observed_value=2020 if "2020" in non_headcount_quote else 27,
            evidence_quote=non_headcount_quote,
        )
        rejected_number = _validated_findings(
            {"findings": [unrelated_number]},
            targets=("headcount",),
            fetched_pages={url: non_headcount_quote},
            first_party_domains={"acme.example"},
            identity_names={"acme"},
        )
        assert rejected_number["headcount"]["status"] == "UNPROVEN"


def test_unproven_rebrand_conflict_is_not_turned_into_a_false_mismatch():
    company = _company(
        name="Wayground formerly Quizizz",
        website="https://quizizz.com",
        linkedin="https://www.linkedin.com/company/quizizz",
    )
    verdict = _complete_verdict(
        observed_company_name="Wayground",
        observed_company_website="https://wayground.com",
        observed_company_linkedin="https://www.linkedin.com/company/quizizz",
    )
    result = _reverify_decision(
        verdict,
        "",
        "",
        icp=_icp(),
        company=company,
        verified_rebrand_identity={"status": "UNPROVEN"},
        company_quality=True,
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert (
        result.details["identity_receipt"]["reason_code"]
        == "rebrand_continuity_unproven"
    )


def test_public_stage_needs_listing_proof_not_labels_or_plans():
    assert _stage_quote_supports_observation(
        "public",
        "CoStar Group common stock is listed on NASDAQ under ticker CSGP.",
    )
    assert not _stage_quote_supports_observation("public", "Company type: Public Company")
    assert not _stage_quote_supports_observation(
        "public", "The company plans an initial public offering next year."
    )


def test_investigation_request_uses_frozen_evaluation_date(monkeypatch):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        arguments = {
            "findings": [
                _finding(
                    "stage",
                    status="UNPROVEN",
                    observed_value=None,
                    evidence_url="",
                    evidence_quote="",
                )
            ]
        }
        return 200, {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "submit_findings",
                            "arguments": json.dumps(arguments),
                        },
                    }]
                }
            }]
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "evaluation_date", lambda: date(2026, 9, 18))
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    input_document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert input_document["evaluation_date"] == "2026-09-18"


def test_full_harness_loop_searches_fetches_and_submits_fetched_quote(monkeypatch):
    url = "https://acme.example/investors"
    quote = "Acme common stock is listed on NASDAQ under ticker ACME."
    reasoning_requests = []
    search_queries = []
    fetched_urls = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        reasoning_requests.append(payload)
        turn = len(reasoning_requests)
        if turn == 1:
            name, arguments = "search_web", {"query": "Acme current stock listing"}
        elif turn == 2:
            name, arguments = "fetch_page", {"url": url}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding("stage", evidence_url=url, evidence_quote=quote)]
            }
        return 200, {
            "choices": [{"message": {"tool_calls": [{
                "id": f"call-{turn}",
                "index": 0,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]}}]
        }

    async def fake_search(_session, query, *, key):
        del key
        search_queries.append(query)
        return {"results": [{"url": url}], "notice": "discovery_only_not_evidence"}

    async def fake_fetch(_session, requested_url):
        fetched_urls.append(requested_url)
        return {"ok": True, "url": requested_url, "text": quote}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["evidence_quote"] == quote
    assert result["usage"] == {
        "reasoning_turns": 3,
        "search_calls": 1,
        "fetch_calls": 1,
    }
    assert search_queries == ["Acme current stock listing"]
    assert fetched_urls == [url]
    assert all(
        request["model"] == investigator.INVESTIGATOR_MODEL
        for request in reasoning_requests
    )
    replayed_call = reasoning_requests[1]["messages"][-2]["tool_calls"][0]
    assert set(replayed_call) == {"id", "type", "function"}
    assert set(replayed_call["function"]) == {"name", "arguments"}
    assert all(
        request["tool_choice"] == "required"
        for request in reasoning_requests
    )


def test_required_tool_turn_does_not_interpret_provider_prose(monkeypatch):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        return 200, {
            "choices": [{"message": {
                "content": (
                    "Acme common stock is listed on NASDAQ under ticker ACME."
                )
            }}]
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    diagnostic = {}

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
        diagnostic=diagnostic,
    ))

    assert requests[0]["tool_choice"] == "required"
    assert result["claims"] == {}
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert diagnostic[VERIFIER_FAILURE_REASON_KEY] == MALFORMED_RESPONSE_FAILURE_REASON


def test_harness_rejects_multiple_tool_calls_as_malformed(monkeypatch):
    async def fake_post_json(_session, _url, *, headers, payload):
        del headers, payload
        call = {
            "id": "call",
            "type": "function",
            "function": {
                "name": "search_web",
                "arguments": json.dumps({"query": "Acme"}),
            },
        }
        return 200, {"choices": [{"message": {"tool_calls": [call, call]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    diagnostic = {}
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        diagnostic=diagnostic,
    ))

    assert result["claims"] == {}
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert diagnostic[VERIFIER_FAILURE_REASON_KEY] == MALFORMED_RESPONSE_FAILURE_REASON


def test_harness_closes_search_budget_and_admission_boundary(monkeypatch):
    reasoning_turns = []
    provider_searches = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        reasoning_turns.append(payload)
        turn = len(reasoning_turns)
        if turn <= 4:
            name, arguments = "search_web", {"query": f"Acme query {turn}"}
        else:
            name, arguments = "submit_findings", {
                "findings": [_finding(
                    "stage",
                    status="UNPROVEN",
                    observed_value=None,
                    evidence_url="",
                    evidence_quote="",
                )]
            }
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{turn}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        provider_searches.append(query)
        return {"results": [], "notice": "discovery_only_not_evidence"}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
    ))
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["reasoning_turns"] == investigator.MAX_REASONING_TURNS
    assert result["usage"]["search_calls"] == investigator.MAX_SEARCH_CALLS
    assert len(provider_searches) == investigator.MAX_SEARCH_CALLS
    assert reasoning_turns[-1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }

    monotonic_values = iter((0.0, investigator.ADMISSION_DEADLINE_SECONDS + 1.0))
    monkeypatch.setattr(
        investigator,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )
    admission_result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
    ))
    assert admission_result["claims"]["stage"]["status"] == "UNPROVEN"
    assert admission_result["usage"]["reasoning_turns"] == 0


def test_embedded_reasoning_provider_error_is_typed_infrastructure(monkeypatch):
    async def fake_post_json(_session, _url, *, headers, payload):
        del headers, payload
        return 200, {"error": {"code": 429, "message": "rate limited"}}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    diagnostic = {}

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
        diagnostic=diagnostic,
    ))

    assert result["claims"] == {}
    assert result["failure_reason"] == PROVIDER_ERROR_FAILURE_REASON
    assert diagnostic[VERIFIER_FAILURE_REASON_KEY] == PROVIDER_ERROR_FAILURE_REASON


def test_only_the_lab_scorer_activates_the_investigator_by_default():
    assert CompetitionCompanyScorer().evidence_investigator is False
    scorer = arena_scoring.lab_scorer(
        arena_scoring.build_scorer_policy(
            scoring_adapter_version="qualification_contacts_v3"
        )
    )
    assert scorer.evidence_investigator is True
    assert scorer.company_quality is False
