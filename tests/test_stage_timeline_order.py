from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator


ROUND_URL = "https://acme.example/news/completed-series-b"
ROUND_QUOTE = "Acme completed a $20 million Series B funding round."
TIMELINE_URL = "https://acme.example/investors"
LATER_URL = "https://acme.example/news/later-transaction"


def _finding(status, value=None, url="", quote=""):
    return {
        "target": "stage", "status": status, "observed_value": value,
        "observed_country": "", "observed_state": "",
        "observed_industry": "", "observed_subindustry": "",
        "activity_role": "unresolved", "evidence_url": url,
        "evidence_quote": quote, "supporting_evidence": [],
        "old_name": "", "new_name": "", "old_domain": "",
        "new_domain": "", "shared_linkedin_slug": "",
        "reason": "Reviewed the company-bound stage timeline and transactions.",
    }


@pytest.mark.parametrize(
    (
        "later_event", "round_date", "later_date", "expected_status",
        "expected_value",
    ),
    [
        ("debt", "2024", "2025", "VERIFIED", "Series B"),
        ("series_c", "2024", "2025", "CONTRADICTED", "Series C+"),
        ("acquired", "2024", "2025", "CONTRADICTED", "Acquired"),
        ("series_c", "2024", "2024", "UNPROVEN", None),
        ("series_c", "2024", "20X5", "UNPROVEN", None),
        (
            "series_c", "March 15, 2024", "October 20, 2024",
            "CONTRADICTED", "Series C+",
        ),
    ],
)
def test_company_timeline_order_reaches_existing_stage_finding_contract(
    monkeypatch, later_event, round_date, later_date, expected_status,
    expected_value,
):
    later_quotes = {
        "debt": "Acme secured a $30 million debt facility from Northstar Finance.",
        "series_c": "Acme completed a $40 million Series C funding round.",
        "acquired": "BuyerCo completed its acquisition of Acme.",
    }
    later_quote = later_quotes[later_event]
    timeline = (
        "Acme funding milestones. 2023 Institutional backing: Acme secured "
        f"investment from Northstar Capital. {round_date} Growth: " +
        ROUND_QUOTE + f" {later_date} Next milestone: {later_quote}"
    )
    if expected_status == "VERIFIED":
        submitted = _finding("VERIFIED", "Series B", ROUND_URL, ROUND_QUOTE)
    elif expected_status == "CONTRADICTED":
        submitted = _finding(
            "CONTRADICTED", expected_value, LATER_URL, later_quote,
        )
    else:
        submitted = _finding("UNPROVEN")
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit-stage", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": [submitted],
            })},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", AsyncMock())
    search = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(investigator, "_search_web", search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",), requested_stage="Series B",
        prior_observations={"submitted_source_urls": [
            TIMELINE_URL, ROUND_URL, LATER_URL,
        ]},
        prefetched_pages={
            TIMELINE_URL: {"final_url": TIMELINE_URL, "text": timeline},
            ROUND_URL: {"final_url": ROUND_URL, "text": ROUND_QUOTE},
            LATER_URL: {"final_url": LATER_URL, "text": later_quote},
        },
        verified_homepage_identity={
            "normalized_name": "Acme", "registrable_dns_domain": "acme.example",
        },
    ))

    assert result["claims"]["stage"]["status"] == expected_status
    assert result["claims"]["stage"]["observed_value"] == expected_value
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 0
    search.assert_awaited_once()
    prompt = " ".join(requests[0]["messages"][0]["content"].split())
    assert "review already loaded first-party timelines" in prompt
    assert "Distinct year headings bound to those events" in prompt
    assert "a later-discovered article is not a later event" in prompt
    assert "date intervals for material competing events overlap" in prompt
    assert "More precise dates or explicit source ordering" in prompt
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["server_current_stage_discovery"]["ok"] is True
    assert next(
        page["text"] for page in document["prefetched_sources"]
        if page["url"] == TIMELINE_URL
    ) == timeline


def test_timeline_cannot_supply_quote_under_another_page_url(monkeypatch):
    timeline_quote = "Acme completed a $20 million Series B funding round."
    article_text = "Acme explored financing options but completed no new round."
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit-stage", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": [_finding(
                    "VERIFIED", "Series B", ROUND_URL, timeline_quote,
                )],
            })},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", AsyncMock())
    monkeypatch.setattr(investigator, "_search_web", AsyncMock(
        return_value={"results": []},
    ))
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",), requested_stage="Series B",
        prior_observations={"submitted_source_urls": [TIMELINE_URL, ROUND_URL]},
        prefetched_pages={
            TIMELINE_URL: {"final_url": TIMELINE_URL, "text": timeline_quote},
            ROUND_URL: {"final_url": ROUND_URL, "text": article_text},
        },
    ))

    assert result.get("claims", {}).get("stage", {}).get("status") != "VERIFIED"
    assert result.get("_validated_stage_finding", {}) == {}
    assert requests
