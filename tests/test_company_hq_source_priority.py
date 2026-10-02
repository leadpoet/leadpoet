"""First-party headquarters priority through the Arena company-fit handoff."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH, COMPANY_FIT_MISMATCH, COMPANY_FIT_UNAVAILABLE, company_fit_match,
)


def _company(*, name="Sysdig", website="https://sysdig.com/",
             linkedin="https://www.linkedin.com/company/sysdig"):
    return CompanyOutput(
        company_name=name, company_website=website, company_linkedin=linkedin,
        industry="Software", employee_count="11-50", country="United States",
        state="California", intent_signals=[{
            "description": "launched a cloud security capability",
            "source": "news", "url": "https://news.example/sysdig",
            "date": "2026-09-01", "snippet": "Sysdig launched a capability.",
        }],
    )


def _icp(**overrides):
    values = {
        "icp_id": "west-coast", "prompt": "West Coast cloud security vendors",
        "industry": "Software", "sub_industry": "SaaS",
        "employee_count": "11-50", "company_stage": "",
        "country": "United States", "geography": "United States, West Coast",
        "product_service": "software",
    }
    values.update(overrides)
    return ICPPrompt(**values)


def _complete_verdict(**overrides):
    verdict = {
        "observed_company_name": "Sysdig",
        "observed_company_website": "https://sysdig.com/",
        "observed_company_linkedin": "https://www.linkedin.com/company/sysdig",
        "observed_employee_count": "11-50", "employee_size_matches": True,
        "employee_size_evidence_url": "https://www.linkedin.com/company/sysdig",
        "employee_size_evidence_quote": "Sysdig has 11-50 employees.",
        "observed_industry": "Software", "observed_subindustry": "SaaS",
        "industry_matches": True, "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://sysdig.com/platform/secure",
        "industry_evidence_quote": "Sysdig supplies cloud security software.",
        "observed_hq_country": "United States", "observed_hq_state": "California",
        "geography_matches": True,
        "geography_evidence_url": "https://www.linkedin.com/company/sysdig",
        "geography_evidence_quote": (
            "135 Main Street, San Francisco, California 94105, "
            "United States (HQ)"
        ),
        "reason": "verified",
    }
    verdict.update(overrides)
    return verdict


def _finding(target, **overrides):
    finding = {
        "target": target, "status": "VERIFIED", "observed_value": "",
        "observed_country": "", "observed_state": "",
        "observed_industry": "", "observed_subindustry": "",
        "activity_role": "unresolved", "evidence_url": "",
        "evidence_quote": "", "supporting_evidence": [],
        "old_name": "", "new_name": "", "old_domain": "",
        "new_domain": "", "shared_linkedin_slug": "", "reason": "verified",
    }
    finding.update(overrides)
    return finding

_SYSDIG_CONTACT_URL = "https://www.sysdig.com/contact-us"
_SYSDIG_CONTACT_HQ = (
    "### Headquarters\n4000 Center at North Hills St Ste. 420\n"
    "Raleigh, NC 27609"
)


def _homepage_identity():
    return company_fit_match("homepage verified", details={
        "identity": {
            "decision": COMPANY_FIT_MATCH,
            "evidence_source": "company_homepage",
            "observed_name": "sysdig",
            "observed_domain": "sysdig.com",
            "observed_linkedin_slug": "sysdig",
        },
        "verified_homepage_transport_domain": "sysdig.com",
    })


@pytest.mark.parametrize(
    ("profile_quote", "company_quality"),
    [
        (
            "135 Main Street, 21st Floor, San Francisco, California 94105, "
            "US, San Francisco, California, United States (HQ)",
            True,
        ),
        ("San Francisco, California, United States", False),
        (
            "135 Main Street, 21st Floor, San Francisco, California 94105, "
            "US, San Francisco, California, United States (HQ)",
            True,
        ),
    ],
    ids=["accepted_sysdig_7227", "accepted_sysdig_950f", "accepted_sysdig_a134"],
)
def test_positive_profile_hq_reopens_verified_contact_source(
    monkeypatch, profile_quote, company_quality,
):
    company = _company(
        name="Sysdig", website="https://sysdig.com/",
        linkedin="https://www.linkedin.com/company/sysdig",
    )
    icp = _icp(country="United States", geography="United States, West Coast")
    verdict = _complete_verdict(
        observed_company_name="Sysdig",
        observed_company_website="https://sysdig.com/",
        observed_company_linkedin="https://www.linkedin.com/company/sysdig",
        industry_evidence_quote="Sysdig supplies cloud security software.",
        geography_evidence_url="https://www.linkedin.com/company/sysdig",
        geography_evidence_quote=profile_quote,
    )
    homepage = _homepage_identity()
    calls = []

    async def broad(**_kwargs):
        return verdict, ""

    async def keep_observation(observation, *_args, **_kwargs):
        return observation

    async def investigate(**kwargs):
        calls.append(kwargs)
        assert kwargs["targets"] == ("geography",)
        assert kwargs["homepage_navigation_locators"] == ({
            "url": _SYSDIG_CONTACT_URL, "label": "Contact Us",
        },)
        return {
            "claims": {"geography": _finding(
                "geography", status="CONTRADICTED",
                observed_value="North Carolina, United States",
                observed_country="United States",
                observed_state="North Carolina",
                evidence_url=_SYSDIG_CONTACT_URL,
                evidence_quote=_SYSDIG_CONTACT_HQ,
            )},
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                _SYSDIG_CONTACT_URL: {
                    "final_url": _SYSDIG_CONTACT_URL,
                    "text": _SYSDIG_CONTACT_HQ,
                },
            },
            "failure_reason": "",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation",
        keep_observation,
    )
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    result = asyncio.run(lead_scorer._llm_reverify_company(
        company, icp, require_company_fit_dimensions=True,
        verified_homepage_identity=homepage,
        verified_homepage_navigation_locators=({
            "url": _SYSDIG_CONTACT_URL, "label": "Contact Us",
        },),
        company_quality=company_quality,
        evidence_investigator=True,
    ))
    assert len(calls) == 1
    assert result.details["dimension_decisions"]["geography"] == COMPANY_FIT_MISMATCH
    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["dimension_evidence"]["geography"]["url"] == _SYSDIG_CONTACT_URL


@pytest.mark.parametrize(
    ("source_url", "source_quote", "locators", "expected", "investigate"),
    [
        (
            _SYSDIG_CONTACT_URL, _SYSDIG_CONTACT_HQ,
            ({"url": _SYSDIG_CONTACT_URL, "label": "Contact Us"},),
            COMPANY_FIT_MISMATCH, False,
        ),
        (
            "https://www.linkedin.com/company/sysdig",
            "Sysdig headquarters: San Francisco, California, United States",
            (), COMPANY_FIT_MATCH, False,
        ),
        (
            "https://www.linkedin.com/company/sysdig",
            "Sysdig headquarters: San Francisco, California, United States",
            ({"url": "https://unrelated.example/contact-us", "label": "Contact"},),
            COMPANY_FIT_MATCH, False,
        ),
        (
            "https://www.linkedin.com/company/sysdig",
            "Sysdig headquarters: San Francisco, California, United States",
            ({"url": _SYSDIG_CONTACT_URL, "label": "Contact Us"},),
            COMPANY_FIT_MATCH, True,
        ),
    ],
    ids=["existing_first_party_negative", "linkedin_only_fallback",
         "foreign_contact_not_bound", "first_party_silent_fallback"],
)
def test_first_party_hq_review_preserves_supported_fallbacks(
    monkeypatch, source_url, source_quote, locators, expected, investigate,
):
    state = "North Carolina" if expected == COMPANY_FIT_MISMATCH else "California"
    verdict = _complete_verdict(
        observed_hq_state=state,
        geography_evidence_url=source_url,
        geography_evidence_quote=source_quote,
    )
    calls = []

    async def broad(**_kwargs):
        return verdict, ""

    async def keep_observation(observation, *_args, **_kwargs):
        return observation

    async def bounded(**kwargs):
        calls.append(kwargs)
        return {
            "claims": {"geography": _finding(
                "geography", status="UNPROVEN", reason="first-party page silent",
            )},
            investigator.PRIVATE_FETCHED_PAGES_KEY: {
                _SYSDIG_CONTACT_URL: {
                    "final_url": _SYSDIG_CONTACT_URL,
                    "text": "Contact Sysdig sales. Regional offices are listed below.",
                },
            },
            "failure_reason": "",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation",
        keep_observation,
    )
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", bounded)
    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(), _icp(), require_company_fit_dimensions=True,
        verified_homepage_identity=_homepage_identity(),
        verified_homepage_navigation_locators=locators,
        company_quality=True, evidence_investigator=True,
    ))
    assert result.decision == expected
    assert result.details["dimension_decisions"]["geography"] == expected
    assert bool(calls) is investigate
    if investigate:
        assert calls[0]["targets"] == ("geography",)


@pytest.mark.parametrize(
    ("contact_text", "state", "expected"),
    [
        (_SYSDIG_CONTACT_HQ, "North Carolina", COMPANY_FIT_MISMATCH),
        (
            "### Headquarters\n135 Main Street\nSan Francisco, CA 94105",
            "California", COMPANY_FIT_MATCH,
        ),
        ("Our Raleigh regional office serves East Coast customers.", "", COMPANY_FIT_MATCH),
        ("Contact Sysdig sales. No headquarters listed.", "", COMPANY_FIT_MATCH),
    ],
    ids=["raleigh_first_party", "san_francisco_first_party_agrees",
         "regional_office_is_not_hq", "first_party_silent"],
)
def test_real_bounded_investigator_fetches_contact_before_profile_fallback(
    monkeypatch, contact_text, state, expected,
):
    verdict = _complete_verdict()
    fetched = []
    judged = []

    async def broad(**_kwargs):
        return verdict, ""

    async def keep_observation(observation, *_args, **_kwargs):
        return observation

    async def fetch(_session, url, **_kwargs):
        fetched.append(url)
        assert url == _SYSDIG_CONTACT_URL
        return {"ok": True, "url": url, "final_url": url, "text": contact_text}

    async def judge(_session, _url, *, headers, payload):
        del headers
        judged.append(payload)
        document = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
        assert document["requested_targets"] == ["geography"]
        assert document["server_priority_submitted_source"]["url"] == _SYSDIG_CONTACT_URL
        if isinstance(payload["tool_choice"], dict) and (
            payload["tool_choice"]["function"]["name"] == "search_web"
        ):
            name, arguments = "search_web", {"query": "Sysdig current headquarters"}
        else:
            name = "submit_findings"
            finding = _finding(
                "geography", status="VERIFIED" if state else "UNPROVEN",
                observed_value=f"{state}, United States" if state else None,
                observed_country="United States" if state else "",
                observed_state=state,
                evidence_url=_SYSDIG_CONTACT_URL if state else "",
                evidence_quote=contact_text if state else "",
            )
            arguments = {"findings": [finding]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "hq-1", "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation",
        keep_observation,
    )
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_post_json", judge)
    monkeypatch.setattr(
        investigator, "_search_web", AsyncMock(return_value={"results": []}),
    )
    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(), _icp(), require_company_fit_dimensions=True,
        verified_homepage_identity=_homepage_identity(),
        verified_homepage_navigation_locators=(
            {"url": "https://other.example/contact", "label": "Contact"},
            {"url": _SYSDIG_CONTACT_URL, "label": "Contact Us"},
        ),
        company_quality=True, evidence_investigator=True,
    ))
    assert result.decision == expected
    assert result.details["dimension_decisions"]["geography"] == expected
    assert fetched == [_SYSDIG_CONTACT_URL]
    assert 1 <= len(judged) <= investigator.MAX_REASONING_TURNS
    assert result.details["investigation_receipt"]["usage"]["fetch_calls"] == 1


@pytest.mark.parametrize(
    ("quote", "priority_url", "final_url", "identity_complete", "expected"),
    [
        (_SYSDIG_CONTACT_HQ, _SYSDIG_CONTACT_URL, _SYSDIG_CONTACT_URL,
         True, "VERIFIED"),
        ("Customer Acme is headquartered in Raleigh, NC 27609",
         _SYSDIG_CONTACT_URL, _SYSDIG_CONTACT_URL, True, "UNPROVEN"),
        ("Find your HQ in Raleigh, NC 27609",
         _SYSDIG_CONTACT_URL, _SYSDIG_CONTACT_URL, True, "UNPROVEN"),
        ("Regional office: Raleigh, NC 27609",
         _SYSDIG_CONTACT_URL, _SYSDIG_CONTACT_URL, True, "UNPROVEN"),
        ("Headquarters: Customer Acme, Raleigh, NC 27609",
         _SYSDIG_CONTACT_URL, _SYSDIG_CONTACT_URL, True, "UNPROVEN"),
        (_SYSDIG_CONTACT_HQ, "https://sysdig.com/customer-story",
         _SYSDIG_CONTACT_URL, True, "UNPROVEN"),
        (_SYSDIG_CONTACT_HQ, "https://sysdig.com/about",
         _SYSDIG_CONTACT_URL, True, "UNPROVEN"),
        (_SYSDIG_CONTACT_HQ, _SYSDIG_CONTACT_URL,
         "https://unrelated.example/contact", True, "UNPROVEN"),
        (_SYSDIG_CONTACT_HQ, _SYSDIG_CONTACT_URL,
         _SYSDIG_CONTACT_URL, False, "UNPROVEN"),
    ],
    ids=["exact_bound_hq", "customer_hq", "generic_marketing_hq",
         "regional_office", "customer_hq_heading", "customer_page_not_priority",
         "not_priority_url", "redirected_off_domain", "incomplete_identity"],
)
def test_unnamed_hq_quote_exception_needs_exact_company_binding(
    quote, priority_url, final_url, identity_complete, expected,
):
    identity = {
        "submitted_name": "Sysdig", "observed_name": "Sysdig",
        "verified_name": "sysdig",
        "submitted_domain": "sysdig.com", "observed_domain": "sysdig.com",
        "verified_domain": "sysdig.com",
        "submitted_linkedin_slug": "sysdig",
        "observed_linkedin_slug": "sysdig",
        "verified_linkedin_slug": "sysdig" if identity_complete else "",
    }
    finding = _finding(
        "geography", status="VERIFIED",
        observed_value="North Carolina, United States",
        observed_country="United States", observed_state="North Carolina",
        evidence_url=_SYSDIG_CONTACT_URL, evidence_quote=quote,
    )
    projected = investigator._validated_findings(
        {"findings": [finding]}, targets=("geography",),
        fetched_pages={_SYSDIG_CONTACT_URL: quote},
        fetched_final_urls={_SYSDIG_CONTACT_URL: final_url},
        first_party_domains={"sysdig.com"}, identity_names={"sysdig"},
        identity_anchor=identity, priority_headquarters_url=priority_url,
    )
    assert projected is not None
    assert projected["geography"]["status"] == expected


def test_public_stage_without_source_still_prefetches_known_hq(monkeypatch):
    fetched = []

    async def fetch(_session, url, **_kwargs):
        fetched.append(url)
        return {"ok": True, "url": url, "final_url": url,
                "text": _SYSDIG_CONTACT_HQ}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(return_value={"results": []}))
    monkeypatch.setattr(investigator, "_post_json", AsyncMock(return_value=(503, {})))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Sysdig", "website": "https://sysdig.com/",
                         "linkedin": "https://www.linkedin.com/company/sysdig"},
        targets=("stage", "geography"), requested_stage="Public",
        requested_geography="United States, West Coast",
        verified_homepage_identity=(
            lead_scorer._verified_homepage_identity_anchor(_homepage_identity())
        ),
        homepage_navigation_locators=({
            "url": _SYSDIG_CONTACT_URL, "label": "Contact Us",
        },),
    ))
    assert fetched == [_SYSDIG_CONTACT_URL]
    assert result["failure_reason"]


def test_first_party_review_without_investigator_claims_fails_closed(monkeypatch):
    async def broad(**_kwargs):
        return _complete_verdict(), ""

    async def keep_observation(observation, *_args, **_kwargs):
        return observation

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(
        lead_scorer, "_refresh_linkedin_employee_size_observation",
        keep_observation,
    )
    monkeypatch.setattr(
        lead_scorer, "investigate_company_evidence",
        AsyncMock(return_value={"claims": {}, "failure_reason": "provider_error"}),
    )
    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(), _icp(), require_company_fit_dimensions=True,
        verified_homepage_identity=_homepage_identity(),
        verified_homepage_navigation_locators=({
            "url": _SYSDIG_CONTACT_URL, "label": "Contact Us",
        },),
        company_quality=True, evidence_investigator=True,
    ))
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["investigation_receipt"]["failure_reason"] == "provider_error"


@pytest.mark.parametrize(
    ("targets", "locator", "expected_priority", "expected_fetch"),
    [
        (("stage", "geography"), _SYSDIG_CONTACT_URL,
         _SYSDIG_CONTACT_URL, "https://sysdig.com/investors"),
        (("geography",), "https://sysdig.com/pricing", "",
         "https://sysdig.com/pricing"),
    ],
    ids=["public_source_precedes_hq", "pricing_is_not_hq"],
)
def test_only_dedicated_hq_locator_grants_unnamed_quote_binding(
    monkeypatch, targets, locator, expected_priority, expected_fetch,
):
    captured = []
    fetched = []
    original = investigator._validated_findings

    def validate(*args, **kwargs):
        captured.append(kwargs["priority_headquarters_url"])
        return original(*args, **kwargs)

    async def fetch(_session, url, **_kwargs):
        fetched.append(url)
        return {"ok": True, "url": url, "final_url": url,
                "text": "Sysdig investor relations page."}

    async def judge(_session, _url, *, headers, payload):
        del headers
        forced = payload.get("tool_choice")
        tool_name = (
            forced["function"]["name"]
            if isinstance(forced, dict) else "submit_findings"
        )
        if tool_name == "search_web":
            arguments = {"query": "Sysdig sysdig.com current public listing"}
        else:
            tool_name = "submit_findings"
            arguments = {"findings": [
                _finding(target, status="UNPROVEN", reason="source silent")
                for target in targets
            ]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "priority-1", "type": "function", "function": {
                "name": tool_name, "arguments": json.dumps(arguments),
            },
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_validated_findings", validate)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_post_json", judge)
    monkeypatch.setattr(
        investigator, "_search_web", AsyncMock(return_value={"results": []}),
    )
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Sysdig", "website": "https://sysdig.com/",
                         "linkedin": "https://www.linkedin.com/company/sysdig"},
        targets=targets,
        requested_stage="Public" if "stage" in targets else "",
        requested_geography="United States, West Coast",
        prior_observations=(
            {"submitted_source_urls": ["https://sysdig.com/investors"],
             "stage_dispute_urls": ["https://sysdig.com/investors"]}
            if "stage" in targets else {}
        ),
        verified_homepage_identity=(
            lead_scorer._verified_homepage_identity_anchor(_homepage_identity())
        ),
        homepage_navigation_locators=({"url": locator, "label": "Contact Us"
                                       if expected_priority else "Pricing"},),
    ))
    assert fetched[0] == expected_fetch if expected_priority else fetched == []
    assert captured and set(captured) == {expected_priority}
    assert set(result["claims"]) == set(targets)
