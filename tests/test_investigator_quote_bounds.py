"""Exact quote packaging controls, with the two saved Lium regression spans.

The mocked model proves the repair contract and hard evidence gates. It does
not measure semantic accuracy; provider-backed comparisons cover that claim.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_investigator import _finding, _icp
from qualification.scoring.lead_scorer import _project_investigator_headcount


LIUM = json.loads(
    (Path(__file__).parent / "fixtures/investigator_lium_headcount.json").read_text()
)
URL = LIUM["url"]
PAGE = LIUM["source_excerpt"]
COMPACT = LIUM["compact_quote"]


def _headcount(quote):
    return _finding(
        "headcount", observed_value="11-50", evidence_url=URL,
        evidence_quote=quote,
    )


def _validate(finding, *, page=PAGE, anchor=None, final_url=URL):
    return investigator._validated_findings(
        {"findings": [finding]}, targets=("headcount",),
        fetched_pages={URL: page}, fetched_final_urls={URL: final_url},
        first_party_domains={"lium.ai"}, identity_names={"lium"},
        identity_anchor=anchor,
    )["headcount"]


@pytest.mark.parametrize("length", LIUM["submitted_quote_lengths"])
def test_lium_oversized_quote_without_verified_profile_identity_requires_repair(length):
    raw = PAGE[:length]
    assert raw.index("Company size") > 2000
    assert "11-50 employees" not in raw[:2000]
    finding = _validate(_headcount(raw))
    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == investigator.OVERSIZED_QUOTE_REASON
    assert finding["evidence_url"] == finding["evidence_quote"] == ""
    # The existing correction context retains the decisive fact beyond 2000.
    context = investigator._source_context_for_quote(raw, PAGE)
    assert "External link for Lium" in context
    assert "Company size" in context and "11-50 employees" in context


def test_oversized_quote_with_valid_prefix_also_requires_repair():
    raw = COMPACT + " " * 2000
    assert investigator._quote_occurs(raw[:2000], raw)
    finding = _validate(_headcount(raw), page=raw)
    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == investigator.OVERSIZED_QUOTE_REASON


def test_compact_lium_company_and_size_span_passes_unchanged_gates():
    assert len(COMPACT) < 2000
    assert COMPACT in PAGE
    finding = _validate(_headcount(COMPACT))
    assert finding["status"] == "VERIFIED"
    assert finding["observed_value"] == "11-50"
    assert finding["evidence_quote"] == COMPACT


@pytest.mark.parametrize("fault", ["wrong_company", "unavailable", "size_only", "forged"])
def test_short_quote_still_requires_company_and_exact_loaded_fact(fault):
    quote, page = COMPACT, PAGE
    if fault == "wrong_company":
        quote = page = COMPACT.replace("Lium", "OtherCo")
    elif fault == "unavailable":
        page = "Sign in to view this company."
    elif fault == "size_only":
        quote = "Company size \n\n 11-50 employees"
    else:
        quote = COMPACT.replace("11-50", "51-200")
    finding = _validate(_headcount(quote), page=page)
    assert finding["status"] == "UNPROVEN"
    assert finding["evidence_url"] == finding["evidence_quote"] == ""


@pytest.mark.parametrize("length", LIUM["submitted_quote_lengths"])
def test_lium_oversized_citation_uses_existing_correction_without_research(
    monkeypatch, length,
):
    requests = []

    async def post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        assert len(requests) <= 2, "quote repair must not add another retry loop"
        if len(requests) == 2:
            feedback = json.loads(payload["messages"][-1]["content"])
            rejected = feedback["rejected_findings"][0]
            assert rejected["reason"] == investigator.OVERSIZED_QUOTE_REASON
            assert "External link for Lium" in rejected["source_context"]
            assert "11-50 employees" in rejected["source_context"]
            assert "no prefix was accepted" in feedback["instruction"]
        quote = PAGE[:length] if len(requests) == 1 else COMPACT
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"quote-{len(requests)}", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps(
                {"findings": [_headcount(quote)]}
            )},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(investigator, "_post_json", post)
    search, fetch = AsyncMock(), AsyncMock()
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Lium", "website": "https://lium.ai/",
                         "linkedin": ""},
        targets=("headcount",), requested_employee_buckets=("11-50",),
        positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [URL]},
        prefetched_pages={URL: {"final_url": URL, "text": PAGE}},
    ))
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["claims"]["headcount"]["status"] == "VERIFIED"
    assert result["claims"]["headcount"]["evidence_quote"] == COMPACT
    assert len(requests) == result["usage"]["reasoning_turns"] == 2
    assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
    search.assert_not_called()
    fetch.assert_not_called()


def test_full_context_prompt_keeps_institution_role_and_criterion_logic():
    prompt = " ".join(investigator._SYSTEM_PROMPT.split())
    assert "requested_icp_prompt is the stored ICP description" in prompt
    assert "A vendor serving institutions does not prove the company itself is an institution" in prompt
    assert "Preserve required roles, industry precedence, and structured fit checks" in prompt
    assert "For an explicit OR, prove one allowed alternative and all shared qualifiers" in prompt
    assert "for AND, prove every conjunct for the same company and product" in prompt
    assert "a pay-once license alone do not prove paid product subscriptions" in prompt
    assert "Each quote must be a continuous span of at most 2000 characters" in prompt
    assert len(investigator._SYSTEM_PROMPT) < investigator.OPENROUTER_MAX_CONTENT_CHARS


def test_narrative_intent_and_event_timing_cannot_determine_company_fit():
    """Davies Turner regressed when partnership timing became an attribute gate.

    This checks the prompt contract; repeated provider comparisons must check
    whether the model follows it on the saved freight/scale evidence.
    """
    prompt = " ".join(investigator._SYSTEM_PROMPT.split())
    assert "use only its company activity and role context" in prompt
    assert (
        "Its intent signals, events, dates, and timing belong to separate intent "
        "checks and must never determine any company-fit finding"
    ) in prompt
    assert "Preserve required roles, industry precedence, and structured fit checks" in prompt
    assert "Keep independent product/service requirements in their own check" in prompt


def test_usage_and_research_do_not_by_themselves_prove_commercial_sales():
    prompt = " ".join(investigator._SYSTEM_PROMPT.split())
    assert (
        "A featured study, named collaboration, usage metric, trial or capability "
        "page alone does not establish a commercial customer or deployment"
    ) in prompt
    assert "source evidence of the commercial relationship or clearly commercial production deployment" in prompt
    assert "same company/product" in prompt
    assert "No public price or invoice is required" in prompt
    assert len(investigator._SYSTEM_PROMPT) < investigator.OPENROUTER_MAX_CONTENT_CHARS


ANCHOR = {
    "submitted_name": "Lium", "submitted_domain": "lium.ai",
    "submitted_linkedin_slug": "liumai",
}


@pytest.mark.parametrize("length", LIUM["submitted_quote_lengths"])
@pytest.mark.parametrize("single_line", [False, True])
def test_identity_bound_lium_original_quote_selects_compact_size_span(length, single_line):
    page = " ".join(PAGE.split()) if single_line else PAGE
    finding = _validate(_headcount(PAGE[:length]), page=page, anchor=ANCHOR)
    quote = finding["evidence_quote"]
    assert finding["status"] == "VERIFIED"
    assert len(quote) < 2000
    assert quote == "Lium Industry Technology, Information and Internet Company size 11-50 employees"
    assert investigator._quote_occurs(quote, PAGE[:length])
    assert investigator._quote_occurs(quote, page)
    assert investigator._quote_names_company(quote, {"lium"})
    assert investigator._quote_supports_headcount(quote, "11-50")
    projected = _project_investigator_headcount(
        {"observed_employee_count": None}, finding,
        icp=_icp(employee_count="11-50"), existing_conflict=False,
    )
    assert projected["observed_employee_count"] == "11-50"
    assert projected["employee_size_matches"] is True
    assert projected["employee_size_evidence_quote"] == quote


def test_real_plain_profile_uses_only_its_own_matching_original_span():
    page = LIUM["plain_source_excerpt"]
    assert "\n" not in page
    assert len(page) == 16_009
    end = page.index("11-50 employees") + len("11-50 employees")
    assert end > 2000
    finding = _validate(_headcount(page[:end]), page=page, anchor=ANCHOR)
    assert finding["status"] == "VERIFIED"
    assert finding["evidence_quote"] in page[:end]
    # Different nav/Markdown renderings are not whitespace-only variants.
    assert _validate(_headcount(PAGE[:2090]), page=page, anchor=ANCHOR)["status"] == "UNPROVEN"


@pytest.mark.parametrize("name,domain,slug,band", [
    ("Meridian Analytics", "meridian.example", "meridian-analytics", "51-200"),
    ("Aster Medical", "aster.example", "aster-medical", "201-500"),
    ("Bright Harbor", "brightharbor.example", "bright-harbor", "1,001-5,000"),
])
@pytest.mark.parametrize("single_line", [False, True])
def test_compact_selection_is_generic_across_company_names_and_size_bands(
    name, domain, slug, band, single_line,
):
    page = (
        f"{name} | LinkedIn\n## About\n" + "Established operations and services. " * 58
        + f"\nWebsite\nhttps://{domain}\nExternal link for {name}\n"
        + f"Industry\nTechnology\nCompany size\n{band} employees\nHeadquarters\nLondon"
    )
    raw = page[:page.index(f"{band} employees") + len(f"{band} employees")]
    assert 2000 < len(raw) <= investigator.PROFILE_MAX_CHARACTERS
    source = " ".join(page.split()) if single_line else page
    url = f"https://www.linkedin.com/company/{slug}"
    names = {"".join(character for character in name.casefold() if character.isalnum())}
    finding = investigator._validated_findings(
        {"findings": [_finding("headcount", observed_value=band, evidence_url=url,
                              evidence_quote=raw)]},
        targets=("headcount",), fetched_pages={url: source},
        fetched_final_urls={url: url}, first_party_domains={domain}, identity_names=names,
        identity_anchor={"submitted_name": name, "submitted_domain": domain,
                         "submitted_linkedin_slug": slug},
    )["headcount"]
    assert finding["status"] == "VERIFIED"
    assert investigator._quote_occurs(finding["evidence_quote"], raw)
    assert investigator._quote_supports_headcount(finding["evidence_quote"], band)


@pytest.mark.parametrize("fault", [
    "wrong_final_slug", "wrong_expected_slug", "wrong_company", "wrong_title", "wrong_website",
    "mixed_profile", "conflicting_size", "conflicting_inline_band", "forged_original", "wrong_observed_size",
    "access_wall", "no_compact_span", "source_too_large", "fact_outside_original",
    "website_outside_original",
])
def test_compact_profile_selection_fails_closed(fault):
    page, raw, anchor, final_url = PAGE, PAGE[:2090], dict(ANCHOR), URL
    finding = _headcount(raw)
    if fault == "wrong_final_slug":
        final_url = "https://www.linkedin.com/company/otherco"
    elif fault == "wrong_expected_slug":
        anchor["submitted_linkedin_slug"] = "otherco"
    elif fault == "wrong_company":
        page = page.replace("Lium", "OtherCo").replace("lium.ai", "other.example")
        raw = page[:page.index("11-50 employees") + len("11-50 employees")]
    elif fault == "wrong_website":
        page, raw = page.replace("lium.ai", "other.example"), raw.replace("lium.ai", "other.example")
    elif fault == "wrong_title":
        page, raw = page.replace("Lium | LinkedIn", "OtherCo | LinkedIn"), raw.replace("Lium | LinkedIn", "OtherCo | LinkedIn")
    elif fault == "mixed_profile":
        page += "\nOtherCo | LinkedIn\nAbout\nCompany size\n51-200 employees"
    elif fault == "conflicting_size":
        page += "\nCompany size\n51-200 employees"
    elif fault == "conflicting_inline_band":
        page = page.replace("11-50 employees", "11-50 employees or 51-200 employees")
        raw = page[:page.index("Headquarters")]
    elif fault == "forged_original":
        raw = raw.replace("complex data usable", "complex data profitable")
    elif fault == "wrong_observed_size":
        finding["observed_value"] = "51-200"
    elif fault == "access_wall":
        page = "Sign in | LinkedIn\nEmail or phone\nPassword\n" + page
    elif fault == "no_compact_span":
        page = "Lium | LinkedIn\nAbout\n" + "Operations. " * 190 + "\nCompany size\n11-50 employees"
        raw = page
    elif fault == "source_too_large":
        page += " " * investigator.MAX_PAGE_CHARACTERS
    elif fault == "fact_outside_original":
        raw = PAGE[:PAGE.index("Company size")]
    elif fault == "website_outside_original":
        page = "Lium | LinkedIn\nAbout\n" + "Operations. " * 170 + "\nCompany size\n11-50 employees"
        raw = page
        page += "\nWebsite\nhttps://other.example"
    finding["evidence_quote"] = raw
    result = _validate(finding, page=page, anchor=anchor, final_url=final_url)
    assert result["status"] == "UNPROVEN"
    assert result["evidence_quote"] == result["evidence_url"] == ""


@pytest.mark.parametrize("length", LIUM["submitted_quote_lengths"])
def test_bound_profile_compaction_settles_in_one_submit_without_research(monkeypatch, length):
    post = AsyncMock(return_value=(200, {"choices": [{"message": {"tool_calls": [{
        "id": "one-submit", "type": "function", "function": {
            "name": "submit_findings", "arguments": json.dumps({"findings": [_headcount(PAGE[:length])]}),
        },
    }]}}]}))
    search, fetch = AsyncMock(), AsyncMock()
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Lium", "website": "https://lium.ai/", "linkedin": URL},
        targets=("headcount",), requested_employee_buckets=("11-50",),
        positive_semantic_review=True,
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={},
        prefetched_pages={URL: {"final_url": URL, "text": PAGE}},
    ))
    assert result["_completed_submit"] is True and result["failure_reason"] == ""
    assert result["claims"]["headcount"]["status"] == "VERIFIED"
    assert result["usage"]["reasoning_turns"] == post.call_count == 1
    assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
    search.assert_not_called()
    fetch.assert_not_called()
