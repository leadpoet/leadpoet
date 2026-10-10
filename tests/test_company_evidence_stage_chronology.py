"""Evidence transport and validation controls, not a simulated model judgment.

The public/synthetic fixtures can also be used for provider-backed chronology
checks. These local tests supply the model finding to verify the unchanged
source, stage, current-discovery, and projection gates around that judgment.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lab_arena import operations
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.lead_scorer import (
    _normalize_company_stage,
    _project_investigator_stage,
)


CASES = json.loads((
    Path(__file__).parent / "fixtures/company_evidence/stage_chronology_cases.json"
).read_text())


def _finding(case):
    proven = case["expected_status"] != "UNPROVEN"
    return {
        "target": "stage",
        "status": case["expected_status"],
        "observed_value": case["expected_stage"],
        "evidence_url": case["pages"][case["evidence_page"]]["url"] if proven else "",
        "evidence_quote": case["evidence_quote"] if proven else "",
        "reason": case["reason"],
    }


def test_search_page_dates_cannot_enter_automatic_or_tool_discovery(monkeypatch):
    case = CASES[1]
    page = case["pages"][0]
    publication_date = "2099-12-31T00:00:00.000Z"
    requests = []
    searches = []

    async def fake_post(_session, url, *, headers, payload):
        del headers
        if url == "https://api.exa.ai/search":
            searches.append(payload)
            return 200, {"results": [{
                "url": page["url"], "title": "Acme funding milestones",
                "publishedDate": publication_date, "dateModified": publication_date,
            }]}
        operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        function = (
            {"name": "search_web", "arguments": json.dumps({"query": "Acme current stage"})}
            if len(requests) == 1
            else {"name": "submit_findings", "arguments": json.dumps({"findings": [_finding(case)]})}
        )
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}", "type": "function", "function": function,
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    fetch = AsyncMock(side_effect=AssertionError("Loaded evidence must not be refetched"))
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",), requested_stage="Series B",
        prior_observations={"submitted_source_urls": [page["url"]]},
        prefetched_pages={page["url"]: {"final_url": page["url"], "text": page["text"]}},
    ))

    assert len(searches) == 2
    assert len(requests) == 2
    initial = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    automatic = initial["server_current_stage_discovery"]["discovery"]
    tool = json.loads(next(
        message["content"] for message in requests[1]["messages"] if message["role"] == "tool"
    ))
    for discovery in (automatic, tool):
        assert discovery["results"] == [{"url": page["url"], "title": "Acme funding milestones"}]
        assert discovery["notice"] == "discovery_only_not_evidence"
    assert publication_date not in json.dumps(requests)
    assert initial["prefetched_sources"] == [page]
    assert "December 2, 2025" in initial["prefetched_sources"][0]["text"]
    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["_validated_stage_finding"]["evidence_quote"] == case["evidence_quote"]
    assert result["usage"]["search_calls"] == investigator.MAX_SEARCH_CALLS == 2
    assert result["usage"]["fetch_calls"] == 0
    assert investigator.MAX_FETCH_CALLS == 3
    fetch.assert_not_called()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_loaded_chronology_and_later_stage_controls_keep_existing_gates(monkeypatch, case):
    requests = []
    finding = _finding(case)

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "stage-finding", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": [finding],
            })},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        assert case["name"] in query
        assert "latest funding round acquisition IPO" in query
        return {"results": [{"url": page["url"]} for page in case["pages"]]}

    fetch = AsyncMock(side_effect=AssertionError("Loaded pages must not be refetched"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    pages = {
        page["url"]: {"final_url": page["url"], "text": page["text"]}
        for page in case["pages"]
    }
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": case["name"], "website": case["website"]},
        targets=("stage",), requested_stage=case["requested_stage"],
        positive_semantic_review=True,
        prior_observations={"submitted_source_urls": list(pages)},
        prefetched_pages=pages,
        verified_homepage_identity={
            "normalized_name": case["name"].casefold(),
            "registrable_dns_domain": case["website"].removeprefix("https://"),
        },
    ))

    assert len(requests) == 1
    assert requests[0]["model"] == investigator.POSITIVE_SEMANTIC_REVIEW_MODEL
    data = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert data["prefetched_sources"] == case["pages"]
    assert data["server_current_stage_discovery"]["ok"] is True
    assert result[investigator.PRIVATE_FETCHED_PAGES_KEY] == pages
    assert result["claims"]["stage"]["status"] == case["expected_status"]
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 0
    fetch.assert_not_called()
    if case["expected_status"] == "UNPROVEN":
        assert result["_validated_stage_finding"] == {}
    else:
        validated = result["_validated_stage_finding"]
        assert validated["evidence_quote"] == case["evidence_quote"]
        projected = _project_investigator_stage(
            {"stage_matches": True}, validated,
            icp_stage=_normalize_company_stage(case["requested_stage"]),
        )
        assert projected["stage_matches"] is (case["expected_status"] == "VERIFIED")
