"""Checked industry sources keep the same independently verified retry identity."""

import asyncio
from copy import deepcopy

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from tests.test_matched_company_retry_source_continuity import (
    IDENTITY, PLATFORM_TEXT, PLATFORM_URL, _company, _icp, _retry_cache, _verdict,
)


PAGES = {
    PLATFORM_URL: {"final_url": PLATFORM_URL, "text": PLATFORM_TEXT},
    "https://unibuddy.com/limitations": {
        "final_url": "https://unibuddy.com/limitations",
        "text": "Unibuddy does not offer campus course delivery.",
    },
}


def _investigate(monkeypatch, cache, homepage_identity, *, company_changes=None,
                 verdict_changes=None, transport_domain="unibuddy.com", company_quality=True,
                 targets=("industry",)):
    findings = investigator._unproven_findings(targets, "Incomplete evidence")
    if "industry" in findings:
        findings["industry"]["_industry_review"] = {
            "schema": "industry_evidence_review_v1", "model": "openai/gpt-6.1-sol",
            "decision": "unavailable", "packet_sha256": "a" * 64,
            "source_urls": list(PAGES), "evidence": [],
        }
    investigation = {
        "claims": findings, "failure_reason": "",
        "_completed_submit": True, "usage": {},
        investigator.PRIVATE_FETCHED_PAGES_KEY: deepcopy(PAGES),
    }
    seen = []

    async def investigated(**kwargs):
        seen.append(kwargs)
        return deepcopy(investigation)

    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigated)
    company = _company().model_copy(update=company_changes or {})
    result = asyncio.run(lead_scorer._run_targeted_company_evidence_investigation(
        company=company, icp=_icp(),
        verdict={**_verdict(), **(verdict_changes or {})},
        investigation_targets=targets,
        icp_attribute=_icp().required_attribute, icp_stage="",
        verified_identity=homepage_identity,
        verified_transport_domain=transport_domain,
        structured_employee_size_evidence=None,
        structured_public_company_evidence=None,
        employee_size_conflict=False, company_quality=company_quality,
        matched_company_source_sink={}, matched_company_retry_source_cache=cache,
    ))
    return result, seen[0]


def _source_pages(call):
    return {**call["prefetched_pages"], **call.get("retained_industry_pages", {})}


@pytest.mark.parametrize("homepage_available", [True, False])
@pytest.mark.parametrize("company_quality", [True, False])
def test_complete_same_web_identity_keeps_checked_sources(
        monkeypatch, homepage_available, company_quality):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    homepage = dict(IDENTITY) if homepage_available else {}
    cache = {}
    first, _ = _investigate(monkeypatch, cache, homepage, company_quality=company_quality)
    assert first[1].details["identity_decision"] == "match"
    assert lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY) == PAGES
    second, call = _investigate(monkeypatch, cache, homepage, company_quality=company_quality)
    assert _source_pages(call) == PAGES
    # The fallback admits source text only, not homepage identity or a verdict.
    assert call["verified_homepage_identity"] == homepage
    assert second[0]["industry_matches"] is None


@pytest.mark.parametrize("target", ["required_attribute", "stage", "geography", "headcount", "rebrand"])
def test_checked_industry_sources_remain_available_to_other_active_targets(monkeypatch, target):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    cache = {}
    _investigate(monkeypatch, cache, dict(IDENTITY))
    _, call = _investigate(monkeypatch, cache, {}, targets=(target,))
    assert _source_pages(call) == PAGES
    assert call["verified_homepage_identity"] == {}


def test_existing_matched_source_cache_keeps_its_contract_without_industry_policy(monkeypatch):
    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)
    cache = _retry_cache()
    _, call = _investigate(monkeypatch, cache, {})
    assert _source_pages(call) == {PLATFORM_URL: PAGES[PLATFORM_URL]}


@pytest.mark.parametrize("company_changes,verdict_changes,transport_domain", [
    ({"company_name": "Other"}, {"observed_company_name": "Other"}, "unibuddy.com"),
    ({"company_website": "https://other.example/"},
     {"observed_company_website": "https://other.example/"}, "other.example"),
    ({"company_linkedin": "https://www.linkedin.com/company/other"},
     {"observed_company_linkedin": "https://www.linkedin.com/company/other"}, "unibuddy.com"),
])
def test_changed_complete_web_identity_cannot_read_another_cache(
        monkeypatch, company_changes, verdict_changes, transport_domain):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    cache = {}
    _investigate(monkeypatch, cache, dict(IDENTITY))
    _, call = _investigate(monkeypatch, cache, {}, company_changes=company_changes,
                           verdict_changes=verdict_changes, transport_domain=transport_domain)
    assert _source_pages(call) == {}


@pytest.mark.parametrize("verdict_changes", [
    {"observed_company_name": ""},
    {"observed_company_website": ""},
    {"observed_company_linkedin": ""},
    {"observed_company_website": "https://user:password@unibuddy.com/"},
    {"observed_company_linkedin": "https://www.linkedin.com/in/unibuddy"},
    {"observed_company_linkedin": "https://linkedin.com.evil.example/company/unibuddy"},
])
def test_incomplete_or_malformed_web_identity_cannot_read_retry_sources(
        monkeypatch, verdict_changes):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    cache = {}
    _investigate(monkeypatch, cache, dict(IDENTITY))
    before = deepcopy(cache)
    _, call = _investigate(monkeypatch, cache, {}, verdict_changes=verdict_changes)
    assert _source_pages(call) == {}
    assert cache == before


@pytest.mark.parametrize("homepage", [
    {**IDENTITY, "registrable_dns_domain": "other.example"},
    {"normalized_name": "Unibuddy"},
])
def test_existing_conflicting_or_incomplete_homepage_anchor_is_not_bypassed(
        monkeypatch, homepage):
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "openai/gpt-6.1-sol")
    cache = {}
    _investigate(monkeypatch, cache, dict(IDENTITY))
    _, call = _investigate(monkeypatch, cache, homepage)
    assert _source_pages(call) == {}
