from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import lead_scorer
from qualification.scoring.company_evidence_investigator import (
    MAX_FETCH_CALLS,
    MAX_PAGE_CHARACTERS,
)
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
    company_fit_unavailable,
)
from qualification.scoring.competition import CompetitionCompanyScorer
from qualification.scoring.evaluation_clock import use_evaluation_date


PLATFORM_URL = "https://unibuddy.com/platform/"
PLATFORM_QUOTE = (
    "Unibuddy helps higher education institutions manage student ambassador "
    "programs, enrollment journeys, and student communication."
)
PLATFORM_TEXT = f"Unibuddy platform\n{PLATFORM_QUOTE}\nAdmin and CRM workflows."
IDENTITY = {
    "normalized_name": "Unibuddy",
    "registrable_dns_domain": "unibuddy.com",
    "linkedin_company_slug": "unibuddy",
}


def _fit_result(*, decision: str = COMPANY_FIT_MATCH):
    details = {
        "dimension_evidence": {
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "web_identity_receipt": {
                    "decision": COMPANY_FIT_MATCH,
                    "observed_name": "Unibuddy",
                    "observed_domain": "unibuddy.com",
                    "observed_linkedin_slug": "unibuddy",
                    "evidence_source": "company_web_reverification",
                },
            },
            "industry": {
                "decision": COMPANY_FIT_MATCH,
                "observed_decision": COMPANY_FIT_MATCH,
                "web_evidence": {
                    "url": PLATFORM_URL,
                    "quote": PLATFORM_QUOTE,
                },
            },
        },
    }
    if decision == COMPANY_FIT_MATCH:
        return company_fit_match("industry matched", details=details)
    return company_fit_unavailable("another dimension is unresolved", details=details)


def _retry_cache(*, result=None, text: str = PLATFORM_TEXT):
    source_cache = {
        PLATFORM_URL: {"final_url": PLATFORM_URL, "text": text},
    }
    cache = {}
    lead_scorer._retain_matched_company_retry_sources(
        cache,
        source_cache,
        result or _fit_result(),
    )
    return cache


def _company() -> CompanyOutput:
    return CompanyOutput(
        company_name="Unibuddy",
        company_website="https://unibuddy.com/",
        company_linkedin="https://www.linkedin.com/company/unibuddy",
        industry="Education",
        employee_count="201-500",
        country="United Kingdom",
        state="",
        intent_signals=[{
            "description": "Unibuddy announced a new enrollment workflow.",
            "source": "news",
            "url": "https://unibuddy.com/news/enrollment-workflow",
            "date": "2026-09-01",
            "snippet": "The workflow supports university enrollment teams.",
        }],
    )


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="higher-ed",
        prompt="Higher education platforms",
        industry="Education",
        sub_industry="Higher education services",
        employee_count="201-500",
        company_stage="",
        geography="United Kingdom",
        product_service=(
            "A student enrollment, learning, or campus-operations platform "
            "used by education providers to manage admissions, programs, and "
            "learner workflows."
        ),
        required_attribute=(
            "Operates a software or services platform used by education "
            "providers to manage enrollment, learning delivery, student "
            "communication, or campus operations."
        ),
        intent_signals=["Announced a new enrollment workflow"],
    )


def _verdict() -> dict:
    return {
        "observed_company_name": "Unibuddy",
        "observed_company_website": "https://unibuddy.com/",
        "observed_company_linkedin": (
            "https://www.linkedin.com/company/unibuddy"
        ),
        "observed_company_stage": "",
        "stage_matches": None,
        "stage_evidence_url": "",
        "stage_evidence_quote": "",
        "observed_employee_count": "201-500",
        "employee_size_matches": True,
        "employee_size_evidence_url": (
            "https://www.linkedin.com/company/unibuddy"
        ),
        "employee_size_evidence_quote": "Company size 201-500 employees",
        "observed_industry": "Education",
        "observed_subindustry": "Higher education services",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": PLATFORM_URL,
        "industry_evidence_quote": PLATFORM_QUOTE,
        "attribute_satisfied": True,
        "required_attribute_evidence_url": PLATFORM_URL,
        "required_attribute_evidence_quote": PLATFORM_QUOTE,
        "observed_hq_country": "United Kingdom",
        "observed_hq_state": "",
        "geography_matches": True,
        "geography_evidence_url": "https://unibuddy.com/about/",
        "geography_evidence_quote": "Unibuddy is based in London, UK.",
        "reason": "verified",
    }


def _finding(*, status: str, role: str, reason: str) -> dict:
    return {
        "target": "industry",
        "status": status,
        "observed_value": "Higher education enrollment platform",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "Education",
        "observed_subindustry": "Higher education services",
        "activity_role": role,
        "evidence_url": PLATFORM_URL,
        "evidence_quote": PLATFORM_QUOTE,
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": reason,
    }


def test_final_matched_dimension_persists_even_when_another_dimension_is_unresolved():
    cache = _retry_cache(result=_fit_result(decision=COMPANY_FIT_UNAVAILABLE))

    assert cache == {
        "verified_identity": IDENTITY,
        "pages": {
            PLATFORM_URL: {
                "final_url": PLATFORM_URL,
                "text": PLATFORM_TEXT,
            },
        },
    }
    assert lead_scorer._matched_company_retry_prefetched_pages(
        cache, IDENTITY
    ) == cache["pages"]


def test_later_matched_page_does_not_replace_first_retry_page():
    cache = _retry_cache()
    later_url = "https://unibuddy.com/about/"
    later_quote = "Unibuddy is based in London, UK."
    later_text = f"About Unibuddy\n{later_quote}"
    later_result = _fit_result(decision=COMPANY_FIT_UNAVAILABLE)
    later_result.details["dimension_evidence"]["geography"] = {
        "decision": COMPANY_FIT_MATCH,
        "observed_decision": COMPANY_FIT_MATCH,
        "web_evidence": {"url": later_url, "quote": later_quote},
    }

    lead_scorer._retain_matched_company_retry_sources(
        cache,
        {later_url: {"final_url": later_url, "text": later_text}},
        later_result,
    )

    assert list(cache["pages"]) == [PLATFORM_URL, later_url]
    assert cache["pages"][PLATFORM_URL]["text"] == PLATFORM_TEXT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normalized_name", "Other University Platform"),
        ("registrable_dns_domain", "other.example"),
        ("linkedin_company_slug", "other-platform"),
    ],
)
def test_retry_source_does_not_cross_verified_identity(field, value):
    current = {**IDENTITY, field: value}
    assert lead_scorer._matched_company_retry_prefetched_pages(
        _retry_cache(), current
    ) == {}


def test_uncited_unverified_and_tampered_sources_do_not_transfer():
    uncited = "https://unibuddy.com/uncited/"
    source_cache = {
        PLATFORM_URL: {"final_url": PLATFORM_URL, "text": PLATFORM_TEXT},
        uncited: {"final_url": uncited, "text": "Uncited page"},
    }
    cache = {}
    lead_scorer._retain_matched_company_retry_sources(
        cache, source_cache, _fit_result()
    )
    assert set(cache["pages"]) == {PLATFORM_URL}

    unverified = _fit_result()
    unverified.details["dimension_evidence"]["industry"]["decision"] = (
        COMPANY_FIT_UNAVAILABLE
    )
    empty = {}
    lead_scorer._retain_matched_company_retry_sources(
        empty, source_cache, unverified
    )
    assert empty == {}

    tampered = deepcopy(cache)
    tampered["pages"][PLATFORM_URL]["final_url"] = "https://other.example/"
    assert lead_scorer._matched_company_retry_prefetched_pages(
        tampered, IDENTITY
    ) == {}


def test_retry_source_cache_enforces_existing_page_count_and_size_bounds():
    cache = _retry_cache()
    cache["pages"] = {
        f"https://unibuddy.com/page-{index}": {
            "final_url": f"https://unibuddy.com/page-{index}",
            "text": "bounded",
        }
        for index in range(MAX_FETCH_CALLS + 1)
    }
    assert lead_scorer._matched_company_retry_prefetched_pages(
        cache, IDENTITY
    ) == {}

    cache = _retry_cache()
    cache["pages"][PLATFORM_URL]["text"] = "x" * (MAX_PAGE_CHARACTERS + 1)
    assert lead_scorer._matched_company_retry_prefetched_pages(
        cache, IDENTITY
    ) == {}


def test_unibuddy_platform_page_survives_later_source_unavailable_retry(
    monkeypatch,
):
    captured = []

    async def capture(**kwargs):
        captured.append(kwargs)
        return {
            "claims": {},
            "failure_reason": "investigation source unavailable",
            "usage": {"reasoning_turns": 1, "search_calls": 0, "fetch_calls": 0},
        }

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", capture)
    asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=_company(),
        icp=_icp(),
        verdict=_verdict(),
        investigation_targets=("industry",),
        icp_attribute=_icp().required_attribute,
        icp_stage="",
        verified_identity=IDENTITY,
        verified_transport_domain="unibuddy.com",
        structured_employee_size_evidence=None,
        structured_public_company_evidence=None,
        employee_size_conflict=False,
        company_quality=True,
        required_attribute_source_cache={
            PLATFORM_URL: {"status": "source_unavailable"},
        },
        matched_company_retry_source_cache=_retry_cache(),
        review_positive_semantics=True,
    ))

    assert len(captured) == 1
    assert captured[0]["prefetched_pages"] == {
        PLATFORM_URL: {
            "final_url": PLATFORM_URL,
            "text": PLATFORM_TEXT,
        },
    }
    assert captured[0]["prior_observations"]["submitted_source_urls"][0] == (
        PLATFORM_URL
    )
    assert captured[0]["targets"] == ("industry",)


def test_retry_page_is_untrusted_and_fresh_contradiction_still_wins(monkeypatch):
    calls = []

    async def contradict(**kwargs):
        calls.append(kwargs)
        return {
            "claims": {
                "industry": _finding(
                    status="CONTRADICTED",
                    role="customer_user",
                    reason="Fresh evidence establishes customer use only.",
                ),
            },
            "failure_reason": "",
            "usage": {"reasoning_turns": 1, "search_calls": 0, "fetch_calls": 0},
            lead_scorer.PRIVATE_FETCHED_PAGES_KEY: {
                PLATFORM_URL: {
                    "final_url": PLATFORM_URL,
                    "text": PLATFORM_TEXT,
                },
            },
        }

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", contradict)
    prior = lead_scorer._reverify_decision(
        _verdict(),
        _icp().required_attribute,
        "",
        icp=_icp(),
        company=_company(),
        verified_homepage_identity=IDENTITY,
        verified_homepage_transport_domain="unibuddy.com",
        company_quality=True,
    )
    projected, result, claims, _, _ = asyncio.run(
        lead_scorer._run_targeted_company_evidence_investigation(
            company=_company(),
            icp=_icp(),
            verdict=_verdict(),
            investigation_targets=("industry",),
            icp_attribute=_icp().required_attribute,
            icp_stage="",
            verified_identity=IDENTITY,
            verified_transport_domain="unibuddy.com",
            structured_employee_size_evidence=None,
            structured_public_company_evidence=None,
            employee_size_conflict=False,
            company_quality=True,
            prior_result=prior,
            matched_company_retry_source_cache=_retry_cache(),
            review_positive_semantics=True,
        )
    )

    assert len(calls) == 1
    assert calls[0]["prefetched_pages"][PLATFORM_URL]["text"] == PLATFORM_TEXT
    assert claims["industry"]["status"] == "CONTRADICTED"
    assert projected["industry_matches"] is False
    assert result.decision == COMPANY_FIT_MISMATCH


def test_outer_retry_scope_separates_company_icp_date_policy_domain_and_identity():
    company = _company()
    icp = _icp()
    scorer = CompetitionCompanyScorer(
        integrity_policy=True,
        company_quality=True,
        evidence_investigator=True,
        scorer_policy={"scoring_adapter_version": "one"},
    )
    other_policy = CompetitionCompanyScorer(
        integrity_policy=True,
        company_quality=True,
        evidence_investigator=True,
        scorer_policy={"scoring_adapter_version": "two"},
    )
    variants = [
        company.model_copy(update={"company_name": "Other Platform"}),
        company.model_copy(update={"company_website": "https://other.example/"}),
        company.model_copy(update={
            "company_linkedin": "https://www.linkedin.com/company/other-platform"
        }),
    ]

    with use_evaluation_date("2026-09-27"):
        base = scorer._retry_source_scope_key(company, icp)
        assert all(
            scorer._retry_source_scope_key(variant, icp) != base
            for variant in variants
        )
        assert scorer._retry_source_scope_key(
            company,
            icp.model_copy(update={"required_attribute": "Different criterion"}),
        ) != base
        assert other_policy._retry_source_scope_key(company, icp) != base
    with use_evaluation_date("2026-09-28"):
        assert scorer._retry_source_scope_key(company, icp) != base


def test_competition_adapter_reuses_only_the_namespaced_matching_scope(monkeypatch):
    observed_caches = []

    async def score_company(**kwargs):
        observed_caches.append(kwargs["matched_company_retry_source_cache"])
        return {"final_score": 0.0, "failure_reason": "test-only"}

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    scorer = CompetitionCompanyScorer(
        integrity_policy=True,
        company_quality=True,
        evidence_investigator=True,
        scorer_policy={"scoring_adapter_version": "one"},
    )
    scope = {}
    company = {
        "company_name": "Unibuddy",
        "company_website": "https://unibuddy.com/",
        "company_linkedin": "https://www.linkedin.com/company/unibuddy",
        "industry": "Education",
        "employee_count": "201-500",
        "company_stage": "",
        "country": "United Kingdom",
        "state": "",
        "fit_summary": "Unibuddy supplies higher-education software.",
        "fit_evidence_urls": [PLATFORM_URL],
        "intent_signals": [{
            "matched_icp_signal": 0,
            "description": "Unibuddy announced a new enrollment workflow.",
            "date": "2026-09-01",
            "why_now": "The workflow is a current product signal.",
            "url": "https://unibuddy.com/news/enrollment-workflow",
            "snippet": "The workflow supports university enrollment teams.",
        }],
    }
    icp = _icp().model_dump(mode="json")

    asyncio.run(scorer.score_with_breakdowns(
        [company], icp, False, retry_evidence_scope=scope
    ))
    asyncio.run(scorer.score_with_breakdowns(
        [company], icp, False, retry_evidence_scope=scope
    ))
    changed = {**company, "company_website": "https://other.example/"}
    asyncio.run(scorer.score_with_breakdowns(
        [changed], icp, False, retry_evidence_scope=scope
    ))

    assert observed_caches[0] is observed_caches[1]
    assert observed_caches[2] is not observed_caches[0]
    assert len([
        key for key in scope if key.startswith("matched-company-source-v1:")
    ]) == 2
