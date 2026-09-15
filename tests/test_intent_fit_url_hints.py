"""Current-format event links remain discovery hints, never company-fit proof."""

import asyncio
import json
import re

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import lead_scorer
from qualification.scoring.competition import _normalized_company
from tests.test_fit_evidence_url_compat import (
    _complete_verdict, _icp, _public_company,
)


def current_company():
    row = _public_company()
    row.pop("fit_summary")
    row.pop("fit_evidence_urls")
    row["intent_details"] = "Acme launched its product. This may support customer growth."
    for signal in row["intent_signals"]:
        signal.pop("snippet")
        signal.pop("why_now")
    return row


@pytest.mark.parametrize("company_quality", [False, True])
def test_current_event_link_reaches_independent_fit_lookup(monkeypatch, company_quality):
    row = current_company()
    projected = _normalized_company(
        row, integrity_policy=True, company_quality=company_quality,
    )
    internal = CompanyOutput.model_validate(projected)
    restored = CompanyOutput.model_validate_json(internal.model_dump_json())
    prompts = []

    async def request(**kwargs):
        prompts.append(kwargs["prompt"])
        return _complete_verdict(), ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", request)
    result = asyncio.run(lead_scorer._llm_reverify_company(
        restored, _icp(), require_company_fit_dimensions=True,
    ))

    assert result.decision == "match"
    assert len(prompts) == 1
    locator = json.loads(re.search(
        r"<untrusted_company_locator>(.*?)</untrusted_company_locator>", prompts[0],
    ).group(1))
    assert locator["untrusted_fit_evidence_urls"] == [row["intent_signals"][0]["url"]]
    assert "untrusted discovery hints only" in prompts[0]
    assert "Independently fetch and verify" in prompts[0]
    assert row["intent_details"] not in prompts[0]
    assert row["intent_signals"][0]["description"] not in prompts[0]
    assert restored.intent_signals[0].url == row["intent_signals"][0]["url"]


def test_current_hints_use_existing_deduplication_and_bound_without_losing_signals():
    row = current_company()
    template = row["intent_signals"][0]
    urls = ["https://acme.example.com/news/one", "https://acme.example.com/news/one",
            "https://acme.example.com/news/two", "https://acme.example.com/news/three",
            "https://acme.example.com/news/four"]
    row["intent_signals"] = [
        {**template, "matched_icp_signal": index % 2, "url": url}
        for index, url in enumerate(urls)
    ]

    projected = _normalized_company(row, integrity_policy=True)

    assert projected["fit_evidence_urls"] == []
    assert lead_scorer._fit_evidence_url_hints(CompanyOutput(**projected)) == [
        urls[0], urls[2], urls[4]
    ]
    assert len(projected["intent_signals"]) == 5
    assert {item["url"] for item in projected["intent_signals"]} == set(urls)


def test_current_hint_cannot_override_independently_observed_wrong_stage(monkeypatch):
    row = current_company()
    internal = CompanyOutput.model_validate(_normalized_company(row, integrity_policy=True))
    calls = []

    async def request(**kwargs):
        calls.append(kwargs["prompt"])
        verdict = _complete_verdict()
        verdict.update(observed_company_stage="Series B", stage_matches=False,
                       stage_evidence_quote="Acme completed its Series B funding round.")
        return verdict, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", request)

    result = asyncio.run(lead_scorer._llm_reverify_company(
        internal, _icp(), require_company_fit_dimensions=True,
    ))

    assert result.decision == "mismatch"
    assert len(calls) == 1
