"""A retry may reuse checked source text without inheriting a fit verdict."""
import asyncio
from copy import deepcopy

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from tests.test_matched_company_retry_source_continuity import (
    IDENTITY, PLATFORM_URL, PLATFORM_TEXT, _company, _icp, _verdict, _fit_result,
)


def run_investigation(monkeypatch, *, pages=None, review_changes=None,
                      investigation_changes=None, cache=None, verdict_changes=None,
                      identity_changes=None, navigation=()):
    pages = pages or {
        PLATFORM_URL: {"final_url": PLATFORM_URL, "text": PLATFORM_TEXT},
        "https://unibuddy.com/limitations": {
            "final_url": "https://unibuddy.com/limitations",
            "text": "Unibuddy does not provide learning delivery software.",
        },
    }
    review = {
        "schema": "industry_evidence_review_v1",
        "model": "openai/gpt-6.1-sol", "decision": "unavailable",
        "packet_sha256": "a" * 64, "source_urls": list(pages),
        "evidence": [], "missing_requirements": ["Campus operations"],
        **(review_changes or {}),
    }
    finding = investigator._unproven_findings(
        ("industry",), "Incomplete industry evidence",
    )["industry"]
    finding["_industry_review"] = review
    investigation = {
        "claims": {"industry": finding}, "failure_reason": "",
        "_completed_submit": True, "usage": {},
        investigator.PRIVATE_FETCHED_PAGES_KEY: pages,
        **(investigation_changes or {}),
    }
    seen = []

    async def investigated(**kwargs):
        seen.append(kwargs)
        return deepcopy(investigation)

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigated)
    result = asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=_company(), icp=_icp(), verdict={**_verdict(), **(verdict_changes or {})},
        investigation_targets=("industry",), icp_attribute=_icp().required_attribute,
        icp_stage="", verified_identity={**IDENTITY, **(identity_changes or {})},
        verified_transport_domain="unibuddy.com",
        structured_employee_size_evidence=None, structured_public_company_evidence=None,
        employee_size_conflict=False, company_quality=True,
        matched_company_source_sink={}, matched_company_retry_source_cache=cache,
        homepage_navigation_locators=navigation,
    ))
    return result, seen


def test_industry_only_target_keeps_bound_navigation_without_passing_it(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    navigation = [{"url": PLATFORM_URL, "label": "Platform"}]
    result, seen = run_investigation(monkeypatch, cache={}, navigation=navigation)
    assert seen[0]["homepage_navigation_locators"] == navigation
    assert result[0]["industry_matches"] is None


def test_completed_unavailable_review_keeps_both_support_and_counterevidence(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    cache = {}
    first, _ = run_investigation(monkeypatch, cache=cache)
    assert first[0]["industry_matches"] is None
    retained = lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY)
    assert set(retained) == {PLATFORM_URL, "https://unibuddy.com/limitations"}
    second, seen = run_investigation(monkeypatch, cache=cache)
    assert {**seen[0]["prefetched_pages"], **seen[0].get("retained_industry_pages", {})} == retained
    assert second[0]["industry_matches"] is None
    assert second[2]["industry"]["status"] == "UNPROVEN"


@pytest.mark.parametrize("decision", ["match", "mismatch", "unavailable"])
def test_native_unproven_keeps_new_counterevidence_after_a_model_proposal(monkeypatch, decision):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    positive = {PLATFORM_URL: {"final_url": PLATFORM_URL, "text": PLATFORM_TEXT}}
    cache = {}
    run_investigation(monkeypatch, pages=positive, cache=cache)
    contrary_url = "https://unibuddy.com/product-scope"
    contrary = {"final_url": contrary_url,
                "text": "Unibuddy supplies admissions messaging, not campus learning delivery."}
    result, _ = run_investigation(monkeypatch, pages={**positive, contrary_url: contrary}, cache=cache,
        review_changes={"decision": decision, "evidence": [{"url": contrary_url, "quote": contrary["text"]}]})
    assert result[2]["industry"]["status"] == "UNPROVEN"
    retained = lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY)
    assert retained[contrary_url] == contrary
    assert result[0]["industry_matches"] is None
    _, next_call = run_investigation(monkeypatch, cache=cache)
    assert next_call[0]["prefetched_pages"][contrary_url] == contrary


@pytest.mark.parametrize("fault", ["old_policy", "incomplete", "failure", "missing_receipt", "wrong_model"])
def test_incomplete_or_unbound_review_does_not_change_cache(monkeypatch, fault):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    change, review = {}, {}
    if fault == "old_policy":
        monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL")
    elif fault == "incomplete":
        change["_completed_submit"] = False
        review["packet_sha256"] = None  # No completed industry review.
    elif fault == "failure":
        change["failure_reason"] = "provider_error"
        review["packet_sha256"] = None  # Provider failed before industry review.
    elif fault == "missing_receipt":
        review["schema"] = "not_a_host_receipt"
    else:
        review["model"] = "another/model"
    cache = {}
    run_investigation(monkeypatch, cache=cache, review_changes=review,
                      investigation_changes=change)
    assert cache == {}


def test_unavailable_cache_rejects_other_domains_and_remains_identity_bound(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    pages = {
        PLATFORM_URL: {"final_url": PLATFORM_URL, "text": PLATFORM_TEXT},
        "https://other.example/capability": {
            "final_url": "https://other.example/capability", "text": "Other supplier capability",
        },
        "https://unibuddy.com/redirect": {
            "final_url": "https://other.example/capability", "text": "Other supplier capability",
        },
    }
    cache = {}
    run_investigation(monkeypatch, pages=pages, cache=cache)
    assert lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY) == {
        PLATFORM_URL: pages[PLATFORM_URL],
    }
    for field in IDENTITY:
        assert lead_scorer._matched_company_retry_prefetched_pages(
            cache, {**IDENTITY, field: "another-company.example"},
        ) == {}


def test_more_than_shared_prefetch_limit_reaches_industry_without_expanding_it(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    pages = {
        f"https://unibuddy.com/capability/{i}": {
            "final_url": f"https://unibuddy.com/capability/{i}",
            "text": f"Unibuddy supplies capability {i}.",
        } for i in range(7)
    }
    cache = {}
    run_investigation(monkeypatch, pages=pages, cache=cache)
    _, seen = run_investigation(monkeypatch, pages=pages, cache=cache)
    assert seen[0]["retained_industry_pages"] == pages
    assert len(seen[0]["prefetched_pages"]) == investigator.MAX_PREFETCHED_PAGES == 3


def test_existing_retry_cap_and_fresh_source_priority_are_preserved(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    cap = investigator.max_matched_retry_pages()
    old = {
        f"https://unibuddy.com/old/{i}": {
            "final_url": f"https://unibuddy.com/old/{i}", "text": f"Old fact {i}",
        } for i in range(cap)
    }
    cache = {"verified_identity": dict(IDENTITY), "pages": deepcopy(old)}
    fresh = "https://unibuddy.com/new-limitations"
    pages = {**old, fresh: {"final_url": fresh, "text": "Unibuddy no longer supplies this capability."}}
    before = deepcopy(pages)
    cited = next(reversed(old))
    run_investigation(monkeypatch, pages=pages, cache=cache,
                      review_changes={"evidence": [{"url": cited, "quote": old[cited]["text"]}]})
    retained = lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY)
    assert len(retained) == cap
    assert retained[fresh] == pages[fresh]
    assert list(retained)[0] == cited
    assert pages == before


def test_incomplete_independent_identity_cannot_retain_industry_sources(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    cache = {}
    run_investigation(monkeypatch, cache=cache,
                      verdict_changes={"observed_company_linkedin": ""},
                      identity_changes={"linkedin_company_slug": ""})
    assert cache == {}


def test_unresolved_industry_does_not_evict_a_matched_other_dimension(monkeypatch):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    stage_url = "https://unibuddy.com/funding"
    quote = "Unibuddy raised a Series B funding round."
    result = _fit_result()
    result.details["dimension_evidence"].pop("industry")
    result.details["dimension_evidence"]["stage"] = {
        "decision": "match", "web_evidence": {"url": stage_url, "quote": quote},
    }
    cache = {"verified_identity": dict(IDENTITY), "pages": {
        stage_url: {"final_url": stage_url, "text": quote},
    }}
    pages = {
        f"https://unibuddy.com/page/{i}": {
            "final_url": f"https://unibuddy.com/page/{i}", "text": f"Fact {i}",
        } for i in range(investigator.max_matched_retry_pages())
    }
    review = {"schema": "industry_evidence_review_v1", "model": "openai/gpt-6.1-sol",
              "decision": "unavailable", "packet_sha256": "a" * 64,
              "source_urls": list(pages), "evidence": []}
    lead_scorer._retain_unresolved_industry_retry_sources(cache, result, {
        "_completed_submit": True, "failure_reason": "",
        "claims": {"industry": {"status": "UNPROVEN", "_industry_review": review}},
        investigator.PRIVATE_FETCHED_PAGES_KEY: pages,
    })
    retained = lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY)
    assert list(retained)[0] == stage_url
    assert retained[stage_url]["text"] == quote
    assert len(retained) == investigator.max_matched_retry_pages()
