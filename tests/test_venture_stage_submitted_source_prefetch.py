from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator


ROUND_URL = (
    "https://sokin.com/news/"
    "sokin-raises-50m-series-b-following-100-year-on-year-growth"
)
ROUND_QUOTE = (
    "Sokin today announced it has secured $50 million in Series B funding "
    "to accelerate its global expansion and product capabilities."
)


def _finding(target: str, *, status: str = "UNPROVEN") -> dict:
    return {
        "target": target,
        "status": status,
        "observed_value": "Series B" if status == "VERIFIED" else None,
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": ROUND_URL if status == "VERIFIED" else "",
        "evidence_quote": ROUND_QUOTE if status == "VERIFIED" else "",
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "Independently reviewed current stage.",
    }


def _run_priority_case(monkeypatch, *, cached=False, geography=False):
    homepage = "https://sokin.com/"
    investors = "https://sokin.com/investors"
    contact = "https://sokin.com/contact-us"
    requests = []
    fetch = AsyncMock(return_value={
        "ok": True, "url": contact if geography else ROUND_URL,
        "final_url": contact if geography else ROUND_URL,
        "text": "Sokin headquarters: London, United Kingdom." if geography
        else ROUND_QUOTE,
    })

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        findings = [
            _finding("stage", status="UNPROVEN" if geography else "VERIFIED"),
        ]
        if geography:
            findings.append(_finding("geography"))
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": findings,
            })},
        }]}}]}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    search = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(investigator, "_search_web", search)
    pages = {
        homepage: {"final_url": homepage, "text": "Sokin moves money globally."},
        investors: {"final_url": investors, "text": "Sokin investor relations."},
    }
    if cached:
        pages[ROUND_URL] = {"final_url": ROUND_URL, "text": ROUND_QUOTE}
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Sokin", "website": homepage},
        targets=("stage", "geography") if geography else ("stage",),
        requested_stage="Series B",
        requested_geography="United Kingdom" if geography else "",
        prior_observations={
            "submitted_source_urls": [homepage, investors, ROUND_URL],
            "untrusted_company_stage_evidence": [{
                "url": ROUND_URL, "quote": ROUND_QUOTE,
            }],
        },
        verified_homepage_identity={
            "normalized_name": "Sokin",
            "registrable_dns_domain": "sokin.com",
            "linkedin_company_slug": "sokin",
        },
        homepage_navigation_locators=[{
            "url": contact, "label": "Contact us",
        }] if geography else (),
        prefetched_pages=pages,
    ))
    document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    return result, document, fetch, search


@pytest.mark.parametrize("cached", [False, True])
def test_saved_first_party_series_b_source_reaches_current_stage_judge(
    monkeypatch, cached,
):
    result, document, fetch, search = _run_priority_case(
        monkeypatch, cached=cached,
    )

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["_validated_stage_finding"] == result["claims"]["stage"]
    assert document["server_priority_submitted_source"] == {
        "url": ROUND_URL,
        "ok": True,
        "cache_hit": cached,
        "notice": "server_selected_first_party_locator_is_untrusted_evidence",
    }
    assert any(
        source["url"] == ROUND_URL and ROUND_QUOTE in source["text"]
        for source in document["prefetched_sources"]
    )
    assert document["server_current_stage_discovery"]["ok"] is True
    assert document["investigation_limits"]["prefetched_pages"] == (
        3 if cached else 2
    )
    assert result["usage"]["fetch_calls"] == (0 if cached else 1)
    if cached:
        fetch.assert_not_awaited()
    else:
        fetch.assert_awaited_once()
        assert fetch.await_args.args[1] == ROUND_URL
    search.assert_awaited_once()


def test_headquarters_locator_keeps_priority_over_submitted_round(monkeypatch):
    result, document, fetch, search = _run_priority_case(
        monkeypatch, geography=True,
    )

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert document["server_priority_submitted_source"]["url"] == (
        "https://sokin.com/contact-us"
    )
    fetch.assert_awaited_once()
    assert fetch.await_args.args[1] == "https://sokin.com/contact-us"
    search.assert_awaited_once()


@pytest.mark.parametrize("change", [
    "different_company", "different_stage", "other_domain", "disputed_stage",
    "missing_identity",
])
def test_saved_round_priority_rejects_unbound_or_disputed_hint(change):
    quote = ROUND_QUOTE
    url = ROUND_URL
    disputes = ()
    if change == "different_company":
        quote = quote.replace("Sokin", "OtherCo")
    elif change == "different_stage":
        quote = quote.replace("Series B", "Series A")
    elif change == "other_domain":
        url = url.replace("sokin.com", "other.example")
    else:
        disputes = ("https://sokin.com/news/completed-series-c",)

    assert investigator._venture_stage_submitted_source_to_prefetch(
        requested_stage="series b",
        submitted_stage_evidence=[{"url": url, "quote": quote}],
        submitted_stage_source_urls=(url,),
        stage_dispute_urls=disputes,
        first_party_domains={"sokin.com"},
        identity_names=set() if change == "missing_identity" else {"sokin"},
    ) == ""
