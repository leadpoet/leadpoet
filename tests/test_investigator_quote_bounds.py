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
from tests.test_company_evidence_investigator import _finding


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


def _validate(finding, *, page=PAGE):
    return investigator._validated_findings(
        {"findings": [finding]}, targets=("headcount",),
        fetched_pages={URL: page}, fetched_final_urls={URL: URL},
        first_party_domains={"lium.ai"}, identity_names={"lium"},
    )["headcount"]


@pytest.mark.parametrize("length", LIUM["submitted_quote_lengths"])
def test_lium_saved_oversized_quote_is_not_silently_truncated(length):
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
                         "linkedin": URL},
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
