"""Compatibility and trust-boundary tests for public fit-evidence URL hints."""

import asyncio
import json
import re

import pytest
from pydantic import ValidationError

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.competition_models import CompetitionCompany
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
    company_fit_mismatch,
)
from qualification.scoring.competition import _normalized_company
from qualification.scoring.competition import (
    count_penalizable_false_positives,
    scorer_breakdown_has_retryable_infrastructure_failure,
)
from qualification.scoring.lead_scorer import (
    EMPLOYEE_SIZE_VERIFICATION_FAILURE_CLASS,
    INSUFFICIENT_COMPANY_FIT_EVIDENCE_FAILURE_CLASS,
    _fit_evidence_url_hints,
    _llm_reverify_company,
    _verify_company_fit,
    _web_identity_receipt,
)


def _public_company(**updates):
    company = {
        "company_name": "Acme",
        "company_website": "https://acme.example.com",
        "company_linkedin": "https://linkedin.com/company/acme",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "state": "",
        "fit_summary": "Current public sources support the requested fit.",
        "fit_evidence_urls": [
            "https://acme.example.com/about",
            "https://news.example.com/acme-profile",
        ],
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": "Acme announced a product launch.",
                "date": "2026-08-20",
                "why_now": "The launch creates a current operational change.",
                "url": "https://acme.example.com/news/launch",
                "snippet": "Acme launched the product on 20 August 2026.",
            }
        ],
    }
    company.update(updates)
    return company


def _company(**updates):
    values = {
        "company_name": "Acme",
        "company_website": "https://acme.example.com",
        "company_linkedin": "https://linkedin.com/company/acme",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "description": "UNTRUSTED SUMMARY MUST NOT REACH THE VERIFIER",
        "fit_evidence_urls": ["https://acme.example.com/about"],
        "intent_signals": [
            {
                "description": "Acme announced a product launch.",
                "source": "news",
                "url": "https://acme.example.com/news/launch",
                "date": "2026-08-20",
                "snippet": "Acme launched the product on 20 August 2026.",
            }
        ],
    }
    values.update(updates)
    return CompanyOutput(**values)


def _icp():
    return ICPPrompt(
        icp_id="fit-hints",
        prompt="US Series A software companies",
        industry="Software",
        sub_industry="SaaS",
        employee_count="51-200",
        company_stage="Series A",
        geography="United States",
        country="United States",
        product_service="Workflow software",
    )


def _complete_verdict(*, name="Acme", website="https://acme.example.com"):
    verdict = {
        "observed_company_name": name,
        "observed_company_website": website,
        "observed_company_linkedin": (
            "https://linkedin.com/company/acme"
            if name == "Acme"
            else "https://linkedin.com/company/other"
        ),
        "observed_employee_count": "51-200",
        "employee_size_matches": True,
        "observed_industry": "Software",
        "observed_subindustry": "SaaS",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "observed_hq_country": "United States",
        "observed_hq_state": "",
        "geography_matches": True,
        "observed_company_stage": "Series A",
        "stage_matches": True,
        "reason": "Independent public pages support the observed values.",
    }
    for dimension in ("employee_size", "industry", "geography", "stage"):
        verdict[f"{dimension}_evidence_url"] = (
            f"https://independent.example.com/{dimension}"
        )
        verdict[f"{dimension}_evidence_quote"] = (
            "Acme completed its Series A funding round."
            if dimension == "stage"
            else f"Verified {dimension}."
        )
    return verdict


def _verified_homepage_identity(
    *,
    observed_name="acme",
    observed_domain="acme.example.com",
    observed_linkedin_slug="acme",
):
    return company_fit_match(
        "verified from homepage",
        details={
            "identity": {
                "decision": "match",
                "evidence_source": "company_homepage",
                "observed_name": observed_name,
                "observed_domain": observed_domain,
                "observed_linkedin_slug": observed_linkedin_slug,
            }
        },
    )


def test_public_fit_urls_reach_internal_verifier_hints():
    public = CompetitionCompany.model_validate(_public_company()).model_dump(
        mode="json"
    )
    projected = _normalized_company(public)
    assert projected["fit_evidence_urls"] == [
        "https://acme.example.com/about",
        "https://news.example.com/acme-profile",
    ]

    internal = CompanyOutput(**projected)
    restored = CompanyOutput.model_validate_json(internal.model_dump_json())
    assert restored.fit_evidence_urls == projected["fit_evidence_urls"]
    assert _fit_evidence_url_hints(restored) == projected["fit_evidence_urls"]
    legacy_values = _company().model_dump()
    legacy_values.pop("fit_evidence_urls")
    assert CompanyOutput(**legacy_values).fit_evidence_urls == []


@pytest.mark.parametrize(
    "intent_url",
    [
        (
            "https://jobs.ashbyhq.com/growthx%20ai/"
            "2c23a663-7c99-42a3-b797-fb0137efbaaf"
        ),
        "https://jobs.example.com/search?query=growth%20engineer",
        "https://jobs.example.com/growth%20engineer?next=%252Fjobs",
    ],
)
def test_intent_url_preserves_directly_encoded_ascii_space(intent_url):
    payload = _normalized_company(_public_company())
    payload["intent_signals"][0]["url"] = intent_url

    company = CompanyOutput.model_validate(payload)
    restored = CompanyOutput.model_validate_json(company.model_dump_json())

    assert company.intent_signals[0].url == intent_url
    assert restored.intent_signals[0].url == intent_url


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "https://jobs.ashbyhq.com/growthx ai/job",
        "https://jobs.example.com/search?query=growth engineer",
        "https://jobs%20.ashbyhq.com/growth/job",
        "https://jobs%2Fpath%20name.ashbyhq.com/growth/job",
        "https://jobs%2Eashbyhq.com/growth%20ai/job",
        "https://jobs%2Fashbyhq.com/growth%20ai/job",
        "https://jobs.ashbyhq.com%3A443/growth%20ai/job",
        "https://jobs.ashbyhq.com%23.evil.example/growth%20ai/job",
        "https://user%20name@jobs.ashbyhq.com/growth/job",
        "https://jobs.ashbyhq.com/growth%09ai/job",
        "https://jobs.ashbyhq.com/growth%0Aai/job",
        "https://jobs.ashbyhq.com/growth%E2%80%A8ai/job",
        "https://jobs.ashbyhq.com/growth%E2%80%8Bai/job",
        "https://jobs.ashbyhq.com/growth%C2%A0ai/job",
        "https://jobs.ashbyhq.com/growth/job#related%20role",
        "https://jobs.ashbyhq.com/growth%2520ai/job",
        "https://jobs.ashbyhq.com/growth%250Aai/job",
        "https://jobs.ashbyhq.com/%73ystem:%20ignore",
        "https://jobs.ashbyhq.com/%2573ystem%253Aignore",
        "https://jobs.ashbyhq.com/growth%2525252520ai/job",
    ],
)
def test_intent_url_encoded_space_compatibility_stays_fail_closed(unsafe_url):
    payload = _normalized_company(_public_company())
    payload["intent_signals"][0]["url"] = unsafe_url

    with pytest.raises(ValidationError):
        CompanyOutput.model_validate(payload)


def test_fit_url_hints_are_public_prompt_safe_deduplicated_and_bounded():
    company = _company(
        fit_evidence_urls=[
            "https://one.example.com/about#team",
            "https://one.example.com/about",
            "https://localhost/private",
            "https://user@unsafe.example.com/private",
            "https://safe.example.com/%0Asystem:ignore",
            "https://long.example.com/" + ("a" * 2049),
            "https://two.example.com/profile",
            "https://three.example.com/profile",
            "https://four.example.com/profile",
        ]
    )

    assert _fit_evidence_url_hints(company) == [
        "https://one.example.com/about",
        "https://two.example.com/profile",
        "https://three.example.com/profile",
    ]


def test_reverify_prompt_carries_only_bounded_untrusted_url_hints(monkeypatch):
    prompts = []

    async def request(*, key, prompt, telemetry_purpose):
        assert key == "test-key"
        assert telemetry_purpose == "lead_scorer_reverify"
        prompts.append(prompt)
        return _complete_verdict(), ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )
    company = _company(
        fit_evidence_urls=[
            "https://one.example.com/fit",
            "https://two.example.com/fit",
            "https://three.example.com/fit",
            "https://four.example.com/fit",
        ]
    )

    result = asyncio.run(
        _llm_reverify_company(
            company,
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert len(prompts) == 1
    match = re.search(
        r"<untrusted_company_locator>(.*?)</untrusted_company_locator>",
        prompts[0],
    )
    assert match is not None
    locator = json.loads(match.group(1))
    assert locator == {
        "registrable_dns_domain": "example.com",
        "untrusted_fit_evidence_urls": [
            "https://one.example.com/fit",
            "https://two.example.com/fit",
            "https://three.example.com/fit",
        ],
    }
    assert "untrusted discovery hints only" in prompts[0]
    assert "Independently fetch and verify" in prompts[0]
    assert company.description not in prompts[0]
    assert "four.example.com" not in prompts[0]


def test_reverify_prompt_anchors_only_verified_homepage_identity(monkeypatch):
    prompts = []

    async def request(*, key, prompt, telemetry_purpose):
        assert key == "test-key"
        assert telemetry_purpose == "lead_scorer_reverify"
        prompts.append(prompt)
        return _complete_verdict(website="https://example.com"), ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )
    homepage_identity = company_fit_match(
        "verified from homepage",
        details={
            "identity": {
                "decision": "match",
                "evidence_source": "company_homepage",
                "observed_name": "acme",
                "observed_domain": "example.com",
                "observed_linkedin_slug": "acme",
                "submitted_name": "untrusted-submitted-name",
            }
        },
    )

    result = asyncio.run(
        _llm_reverify_company(
            _company(
                company_name="Acme (YC W26)",
                company_website="https://example.com",
            ),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=homepage_identity,
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert len(prompts) == 1
    match = re.search(
        r"<verified_homepage_identity>(.*?)</verified_homepage_identity>",
        prompts[0],
    )
    assert match is not None
    assert json.loads(match.group(1)) == {
        "linkedin_company_slug": "acme",
        "normalized_name": "acme",
        "registrable_dns_domain": "example.com",
    }
    assert "untrusted-submitted-name" not in prompts[0]
    assert "do not substitute a same-name company" in prompts[0]
    assert "not proof of any fit dimension" in prompts[0]


def test_unverified_incomplete_or_oversized_homepage_identity_is_not_anchored(
    monkeypatch,
):
    prompts = []

    async def request(**kwargs):
        prompts.append(kwargs["prompt"])
        return _complete_verdict(), ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )
    receipt = {
        "decision": "match",
        "evidence_source": "company_homepage",
        "observed_name": "acme",
        "observed_domain": "example.com",
        "observed_linkedin_slug": "acme",
    }
    identities = [
        company_fit_match(
            "incomplete",
            details={"identity": {**receipt, "observed_linkedin_slug": ""}},
        ),
        company_fit_match(
            "not homepage-verified",
            details={
                "identity": {
                    **receipt,
                    "evidence_source": "company_web_reverification",
                }
            },
        ),
        company_fit_mismatch(
            "homepage mismatch",
            details={"identity": receipt},
        ),
    ] + [
        company_fit_match(
            f"oversized {field}",
            details={"identity": {**receipt, field: "a" * (limit + 1)}},
        )
        for field, limit in (
            ("observed_name", 200),
            ("observed_domain", 253),
            ("observed_linkedin_slug", 200),
        )
    ]

    for identity in identities:
        result = asyncio.run(
            _llm_reverify_company(
                _company(),
                _icp(),
                require_company_fit_dimensions=True,
                verified_homepage_identity=identity,
            )
        )
        assert result.decision == COMPANY_FIT_MATCH

    assert len(prompts) == len(identities)
    assert all(
        "<verified_homepage_identity>" not in prompt for prompt in prompts
    )


def test_verified_anchor_linkedin_conflict_retries_then_stays_unavailable(
    monkeypatch,
):
    prompts = []

    async def request(**kwargs):
        prompts.append(kwargs["prompt"])
        verdict = _complete_verdict()
        verdict["observed_company_linkedin"] = (
            "https://linkedin.com/company/same-name-other-company"
        )
        return verdict, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_verified_homepage_identity(),
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["identity_receipt"]["reason_code"] == (
        "web_linkedin_conflicts_with_verified_homepage"
    )
    assert result.details["identity_receipt"]["observed_linkedin_slug"] == (
        "same-name-other-company"
    )
    assert len(prompts) == 2
    assert "IDENTITY CONFLICT REPAIR" in prompts[1]
    assert "Do not assume that two different slugs are aliases" in prompts[1]


def test_verified_anchor_linkedin_conflict_can_be_repaired(monkeypatch):
    prompts = []

    async def request(**kwargs):
        prompts.append(kwargs["prompt"])
        verdict = _complete_verdict(
            name="Keppel",
            website="https://www.keppel.com/",
        )
        if len(prompts) == 1:
            verdict["observed_company_linkedin"] = (
                "https://linkedin.com/company/keppel-ltd"
            )
        else:
            verdict["observed_company_linkedin"] = (
                "https://linkedin.com/company/keppel"
            )
        return verdict, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _company(
                company_name="Keppel",
                company_website="https://www.keppel.com/",
                company_linkedin="https://linkedin.com/company/keppel",
            ),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_verified_homepage_identity(
                observed_name=(
                    "keppelglobalassetmanagerandoperatorcreatingsolutions"
                    "forasustainablefuture"
                ),
                observed_domain="keppel.com",
                observed_linkedin_slug="keppel",
            ),
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert len(prompts) == 2
    assert result.details["identity_receipt"]["observed_linkedin_slug"] == (
        "keppel"
    )


def test_web_linkedin_conflict_without_verified_anchor_remains_mismatch(
    monkeypatch,
):
    async def request(**_kwargs):
        verdict = _complete_verdict()
        verdict["observed_company_linkedin"] = (
            "https://linkedin.com/company/acme-ltd"
        )
        return verdict, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["identity_receipt"]["reason_code"] == (
        "identity_mismatch"
    )


def _selsym_company():
    return _company(
        company_name="SelSym Biotech",
        company_website="https://selsym.com/",
        company_linkedin="",
    )


def _selsym_alternate_domain_verdict():
    verdict = _complete_verdict(
        name="SelSym Biotech",
        website="https://selsymbio.com/",
    )
    verdict["observed_company_linkedin"] = (
        "https://linkedin.com/company/selsymbio"
    )
    # Retained SelSym receipt shape: the direct employee source and quote were
    # present, but the observed string did not normalize to a canonical bucket.
    verdict["observed_employee_count"] = "approximately 40 employees"
    verdict["employee_size_matches"] = True
    return verdict


def _selsym_homepage_identity():
    return _verified_homepage_identity(
        observed_name="selsymbiotech",
        observed_domain="selsym.com",
        observed_linkedin_slug="selsymbio",
    )


def test_verified_homepage_same_name_domain_conflict_repairs_then_is_unproven(
    monkeypatch,
):
    prompts = []

    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def homepage(*_args, **_kwargs):
        return _selsym_homepage_identity()

    async def request(**kwargs):
        prompts.append(kwargs["prompt"])
        return _selsym_alternate_domain_verdict(), ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer.run_company_zero_checks", prechecks
    )
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer.verify_company_exists", homepage
    )
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _verify_company_fit(
            _selsym_company(),
            _icp(),
            0.0,
            1.0,
            set(),
            require_https_transport=True,
        )
    )
    identity = result.details["dimension_evidence"]["identity"]
    receipt = identity["web_identity_receipt"]
    breakdown = {
        "final_score": 0.0,
        "failure_reason": f"Company fit unavailable: {result.reason}",
        "verifier_gate_receipts": [result.receipt("company_fit")],
    }

    assert len(prompts) == 2
    assert "prior observed_company_website" in prompts[1]
    assert "Do not assume that the two domains are aliases" in prompts[1]
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == (
        INSUFFICIENT_COMPANY_FIT_EVIDENCE_FAILURE_CLASS
    )
    assert receipt["reason_code"] == (
        "web_domain_conflicts_with_verified_homepage"
    )
    assert receipt["submitted_name"] == receipt["observed_name"] == (
        "selsymbiotech"
    )
    assert receipt["submitted_domain"] == "selsym.com"
    assert receipt["observed_domain"] == "selsymbio.com"
    assert receipt["observed_linkedin_slug"] == "selsymbio"
    assert count_penalizable_false_positives(
        [breakdown], icp_has_intent_signals=True
    ) == (0, 0)
    assert not scorer_breakdown_has_retryable_infrastructure_failure(breakdown)


def test_verified_homepage_same_name_domain_conflict_can_be_repaired(
    monkeypatch,
):
    calls = []

    async def request(**kwargs):
        calls.append(kwargs["telemetry_purpose"])
        if len(calls) == 1:
            return _selsym_alternate_domain_verdict(), ""
        repaired = _complete_verdict(
            name="SelSym Biotech",
            website="https://selsym.com/",
        )
        repaired["observed_company_linkedin"] = (
            "https://linkedin.com/company/selsymbio"
        )
        return repaired, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _selsym_company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_selsym_homepage_identity(),
        )
    )

    assert calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["identity_receipt"]["observed_domain"] == (
        "selsym.com"
    )


def test_verified_homepage_different_domain_and_linkedin_remains_mismatch():
    verdict = _selsym_alternate_domain_verdict()
    verdict["observed_company_linkedin"] = (
        "https://linkedin.com/company/same-name-other-company"
    )

    receipt = _web_identity_receipt(
        _selsym_company(),
        verdict,
        verified_homepage_identity={
            "normalized_name": "selsymbiotech",
            "registrable_dns_domain": "selsym.com",
            "linkedin_company_slug": "selsymbio",
        },
    )

    assert receipt["decision"] == COMPANY_FIT_MISMATCH
    assert receipt["reason_code"] == "identity_mismatch"


def test_same_name_wrong_domain_without_homepage_anchor_remains_mismatch(
    monkeypatch,
):
    calls = []

    async def request(**kwargs):
        calls.append(kwargs["telemetry_purpose"])
        verdict = _complete_verdict(
            name="Global South Utilities",
            website="https://alternate-utilities.example/",
        )
        verdict["observed_company_linkedin"] = (
            "https://linkedin.com/company/global-south-utilities"
        )
        return verdict, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )
    company = _company(
        company_name="Global South Utilities",
        company_website="https://globalsouthutilities.example/",
        company_linkedin="",
    )

    result = asyncio.run(
        _llm_reverify_company(
            company,
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert calls == ["lead_scorer_reverify"]
    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["identity_receipt"]["reason_code"] == (
        "identity_mismatch"
    )


def test_verified_homepage_domain_conflict_repair_failure_remains_retryable(
    monkeypatch,
):
    calls = []

    async def request(**kwargs):
        calls.append(kwargs["telemetry_purpose"])
        if len(calls) == 1:
            return _selsym_alternate_domain_verdict(), ""
        return None, "provider HTTP 503"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _selsym_company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_selsym_homepage_identity(),
        )
    )
    breakdown = {
        "final_score": 0.0,
        "failure_reason": f"Company fit unavailable: {result.reason}",
        "verifier_gate_receipts": [result.receipt("company_fit")],
    }

    assert calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert "failure_class" not in result.details
    assert scorer_breakdown_has_retryable_infrastructure_failure(breakdown)


def test_homepage_domain_conflict_with_failed_size_fetch_remains_retryable(
    monkeypatch,
):
    calls = []
    profile_calls = []

    async def request(**kwargs):
        calls.append(kwargs["telemetry_purpose"])
        verdict = _selsym_alternate_domain_verdict()
        verdict["employee_size_evidence_url"] = (
            "https://linkedin.com/company/selsymbio"
        )
        return verdict, ""

    async def failed_profile_fetch(url):
        profile_calls.append(url)
        return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer.fetch_current_linkedin_company_size",
        failed_profile_fetch,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _selsym_company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_selsym_homepage_identity(),
        )
    )
    breakdown = {
        "final_score": 0.0,
        "failure_reason": f"Company fit unavailable: {result.reason}",
        "verifier_gate_receipts": [result.receipt("company_fit")],
    }

    assert calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert profile_calls == ["https://www.linkedin.com/company/selsymbio"]
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == (
        EMPLOYEE_SIZE_VERIFICATION_FAILURE_CLASS
    )
    assert scorer_breakdown_has_retryable_infrastructure_failure(breakdown)


def test_verified_anchor_does_not_hide_a_different_web_entity(monkeypatch):
    async def request(**_kwargs):
        return _complete_verdict(
            name="Other Company",
            website="https://other.example.com",
        ), ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_verified_homepage_identity(),
        )
    )

    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["identity_receipt"]["reason_code"] == (
        "identity_mismatch"
    )


def test_identity_conflict_repair_preserves_a_real_dimension_mismatch(
    monkeypatch,
):
    prompts = []

    async def request(**kwargs):
        prompts.append(kwargs["prompt"])
        verdict = _complete_verdict()
        verdict.update(
            observed_company_linkedin=(
                "https://linkedin.com/company/acme-ltd"
            ),
            observed_employee_count="201-500",
            employee_size_matches=False,
        )
        return verdict, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_verified_homepage_identity(),
        )
    )

    assert len(prompts) == 2
    assert "IDENTITY CONFLICT REPAIR" in prompts[1]
    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_MISMATCH
    )
    assert result.details["identity_decision"] == COMPANY_FIT_UNAVAILABLE


def test_fit_url_alone_cannot_replace_independent_dimension_proof(monkeypatch):
    calls = []
    incomplete = _complete_verdict()
    for dimension in ("employee_size", "industry", "geography", "stage"):
        incomplete.pop(f"{dimension}_evidence_url")
        incomplete.pop(f"{dimension}_evidence_quote")

    async def request(**kwargs):
        calls.append(kwargs)
        return incomplete, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert len(calls) == 2


def test_wrong_web_identity_still_fails_with_submitted_fit_url(monkeypatch):
    async def request(**_kwargs):
        return _complete_verdict(
            name="Other Company",
            website="https://other.example.com",
        ), ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._request_company_reverify_json",
        request,
    )

    result = asyncio.run(
        _llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["identity_decision"] == COMPANY_FIT_MISMATCH


def test_homepage_identity_mismatch_remains_terminal_before_hint_use(monkeypatch):
    async def prechecks(*_args, **_kwargs):
        return company_fit_match()

    async def homepage(*_args, **_kwargs):
        return company_fit_mismatch("homepage proves a different company")

    async def must_not_reverify(*_args, **_kwargs):
        raise AssertionError("fit URL hints cannot override a homepage mismatch")

    monkeypatch.setattr(
        "qualification.scoring.lead_scorer.run_company_zero_checks", prechecks
    )
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer.verify_company_exists", homepage
    )
    monkeypatch.setattr(
        "qualification.scoring.lead_scorer._llm_reverify_company",
        must_not_reverify,
    )

    result = asyncio.run(
        _verify_company_fit(
            _company(),
            _icp(),
            0.0,
            1.0,
            set(),
            require_https_transport=True,
        )
    )

    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["company_fit_dimensions"]["identity"] == (
        COMPANY_FIT_MISMATCH
    )
