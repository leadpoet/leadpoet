from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.company_verification import _homepage_navigation_locators


DOMAIN = "acme.example"
ROUND_URL = f"https://{DOMAIN}/news/completed-series-b"
OVERVIEW_URL = f"https://{DOMAIN}/investors"
ROUND_QUOTE = "Acme completed a $20 million Series B funding round."
TIMELINE = (
    "Acme funding milestones. 2023 Acme secured institutional investment. "
    "2024 Acme completed a $20 million Series B funding round. "
    "2025 Acme secured a debt facility for expansion."
)
IDENTITY = {
    "normalized_name": "Acme",
    "registrable_dns_domain": DOMAIN,
    "linkedin_company_slug": "acme",
}


def _finding(status):
    return {
        "target": "stage", "status": status,
        "observed_value": "Series B" if status == "VERIFIED" else None,
        "observed_country": "", "observed_state": "",
        "observed_industry": "", "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": ROUND_URL if status == "VERIFIED" else "",
        "evidence_quote": ROUND_QUOTE if status == "VERIFIED" else "",
        "supporting_evidence": [], "old_name": "", "new_name": "",
        "old_domain": "", "new_domain": "", "shared_linkedin_slug": "",
        "reason": "Reviewed submitted evidence and current stage discovery.",
    }


@pytest.mark.parametrize("path", [
    "/investors", "/investor-relations", "/funding-history",
])
def test_exact_submitted_investor_overview_paths_are_discovery_only(path):
    url = f"https://{DOMAIN}{path}"
    assert investigator._submitted_venture_overview_url(
        company_name="Acme", submitted_source_urls=(url,),
        stage_dispute_urls=(), verified_identity=IDENTITY,
    ) == url


def test_submitted_overview_precedes_observed_navigation():
    assert investigator._submitted_venture_overview_url(
        company_name="Acme", submitted_source_urls=(OVERVIEW_URL,),
        stage_dispute_urls=(), verified_identity=IDENTITY,
        homepage_navigation_locators=({
            "url": f"https://{DOMAIN}/funding-history", "label": "Funding",
        },),
    ) == OVERVIEW_URL


def test_saved_series_b_source_has_no_overview_but_observed_homepage_does():
    homepage = "https://sokin.com/"
    article = (
        "https://sokin.com/news/"
        "sokin-raises-50m-series-b-following-100-year-on-year-growth"
    )
    investors = "https://sokin.com/investors"
    locators = _homepage_navigation_locators(
        '<a href="/contact-sales">Contact Sales</a>'
        '<a href="/investors">Learn about our investors</a>',
        final_url=homepage, verified_domain="sokin.com",
    )
    assert investigator._submitted_venture_overview_url(
        company_name="Sokin", submitted_source_urls=(article, homepage),
        stage_dispute_urls=(),
        verified_identity={
            "normalized_name": "Sokin",
            "registrable_dns_domain": "sokin.com",
            "linkedin_company_slug": "sokin",
        },
        homepage_navigation_locators=locators,
    ) == investors


@pytest.mark.parametrize("change", [
    "other_domain", "news_path", "query", "wrong_name", "missing_name",
    "missing_domain", "missing_slug", "dispute",
])
def test_unbound_or_disputed_overview_is_not_selected(change):
    url = OVERVIEW_URL
    identity = dict(IDENTITY)
    disputes = ()
    if change == "other_domain":
        url = "https://other.example/investors"
    elif change == "news_path":
        url = f"https://{DOMAIN}/news/investors"
    elif change == "query":
        url += "?next=https://other.example"
    elif change == "wrong_name":
        identity["normalized_name"] = "OtherCo"
    elif change == "missing_name":
        identity.pop("normalized_name")
    elif change == "missing_domain":
        identity.pop("registrable_dns_domain")
    elif change == "missing_slug":
        identity.pop("linkedin_company_slug")
    else:
        disputes = (f"https://{DOMAIN}/news/later-series-c",)
    assert investigator._submitted_venture_overview_url(
        company_name="Acme", submitted_source_urls=(url,),
        stage_dispute_urls=disputes, verified_identity=identity,
    ) == ""


@pytest.mark.parametrize("mode", ["fresh", "cached", "failed", "no_round", "hq"])
def test_investor_overview_reaches_judge_with_existing_fetch_budget(
    monkeypatch, mode,
):
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        status = "VERIFIED" if mode in {"fresh", "cached", "hq"} else "UNPROVEN"
        findings = [_finding(status)]
        if mode == "hq":
            findings.append({**_finding("UNPROVEN"), "target": "geography"})
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": findings,
            })},
        }]}}]}

    fetch = AsyncMock(return_value=(
        {"ok": False, "error": "http_404"} if mode == "failed" else
        {"ok": True, "url": OVERVIEW_URL,
         "final_url": OVERVIEW_URL, "text": TIMELINE}
    ))
    contact_url = f"https://{DOMAIN}/contact-us"
    if mode == "hq":
        fetch.side_effect = [
            {"ok": True, "url": contact_url, "final_url": contact_url,
             "text": "Contact Acme for sales information."},
            {"ok": True, "url": OVERVIEW_URL, "final_url": OVERVIEW_URL,
             "text": TIMELINE},
        ]
    search = AsyncMock(return_value={"results": []})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)
    pages = {} if mode == "no_round" else {
        ROUND_URL: {"final_url": ROUND_URL, "text": ROUND_QUOTE},
    }
    if mode == "cached":
        pages[OVERVIEW_URL] = {"final_url": OVERVIEW_URL, "text": TIMELINE}
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": f"https://{DOMAIN}"},
        targets=("stage", "geography") if mode == "hq" else ("stage",),
        requested_stage="Series B",
        requested_geography="United Kingdom" if mode == "hq" else "",
        homepage_navigation_locators=[
            {"url": contact_url, "label": "Contact us"},
        ] if mode == "hq" else (),
        prior_observations={
            "submitted_source_urls": [ROUND_URL, OVERVIEW_URL],
            **({"untrusted_company_stage_evidence": [{
                "url": ROUND_URL, "quote": ROUND_QUOTE,
            }]} if mode != "no_round" else {}),
        },
        prefetched_pages=pages,
        verified_homepage_identity=IDENTITY,
    ))
    document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )

    assert result["claims"]["stage"]["status"] == (
        "VERIFIED" if mode in {"fresh", "cached", "hq"} else "UNPROVEN"
    )
    assert document["server_current_stage_discovery"]["ok"] is True
    assert document["server_venture_stage_overview_fetch"]["url"] == OVERVIEW_URL
    assert document["server_venture_stage_overview_fetch"]["cache_hit"] == (
        mode == "cached"
    )
    assert result["usage"]["fetch_calls"] == (
        0 if mode == "cached" else 2 if mode == "hq" else 1
    )
    assert document["investigation_limits"]["prefetched_pages"] == (
        0 if mode == "no_round" else 2 if mode == "cached" else 1
    )
    assert document["investigation_limits"]["remaining_fetch_calls"] == (
        3 if mode == "cached" else 1 if mode == "hq" else 2
    )
    assert any(
        source["url"] == OVERVIEW_URL and source["text"] == TIMELINE
        for source in document.get("prefetched_sources", [])
    ) == (mode != "failed")
    if mode == "cached":
        fetch.assert_not_awaited()
    elif mode == "hq":
        assert [call.args[1] for call in fetch.await_args_list] == [
            contact_url, OVERVIEW_URL,
        ]
        assert document["server_priority_submitted_source"]["url"] == contact_url
    else:
        fetch.assert_awaited_once()
        assert fetch.await_args.args[1] == OVERVIEW_URL
    search.assert_awaited_once()


@pytest.mark.parametrize("change", [
    "valid", "missing_identity", "wrong_domain", "dispute", "wrong_path",
])
def test_observed_homepage_overview_after_hq_priority_uses_two_of_three_calls(
    monkeypatch, change,
):
    contact_url = f"https://{DOMAIN}/contact-us"
    observed_overview = (
        f"https://other.example/investors" if change == "wrong_domain" else
        f"https://{DOMAIN}/news/investors" if change == "wrong_path" else
        OVERVIEW_URL
    )
    raw_homepage = (
        f'<a href="{contact_url}">Contact us</a>'
        f'<a href="{observed_overview}">Learn about our investors</a>'
    )
    locators = _homepage_navigation_locators(
        raw_homepage, final_url=f"https://{DOMAIN}/",
        verified_domain=DOMAIN,
    )
    assert any(item["url"] == contact_url for item in locators)
    identity = {} if change == "missing_identity" else IDENTITY
    disputes = [f"https://{DOMAIN}/news/later-series-c"] if change == "dispute" else []
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": "submit", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({
                "findings": [_finding("UNPROVEN"), {
                    **_finding("UNPROVEN"), "target": "geography",
                }],
            })},
        }]}}]}

    async def fake_fetch(_session, url, **_kwargs):
        return {"ok": True, "url": url, "final_url": url,
                "text": TIMELINE if url == OVERVIEW_URL else "Contact Acme."}

    fetch = AsyncMock(side_effect=fake_fetch)
    search = AsyncMock(return_value={"results": []})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)
    asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": f"https://{DOMAIN}/"},
        targets=("stage", "geography"), requested_stage="Series B",
        requested_geography="United Kingdom",
        prior_observations={
            "submitted_source_urls": [
                ROUND_URL, f"https://{DOMAIN}/", *disputes,
            ],
            "untrusted_company_stage_evidence": [{
                "url": ROUND_URL, "quote": ROUND_QUOTE,
            }],
            **({"stage_dispute_urls": disputes} if disputes else {}),
        },
        prefetched_pages={
            ROUND_URL: {"final_url": ROUND_URL, "text": ROUND_QUOTE},
        },
        homepage_navigation_locators=locators,
        verified_homepage_identity=identity,
    ))
    document = json.loads(
        requests[0]["messages"][1]["content"].split("\n", 1)[1]
    )
    urls = [call.args[1] for call in fetch.await_args_list]
    if change == "valid":
        assert urls == [contact_url, OVERVIEW_URL]
        assert document["server_venture_stage_overview_fetch"]["ok"] is True
        assert document["investigation_limits"]["remaining_fetch_calls"] == 1
        assert any(item["url"] == OVERVIEW_URL for item in document["prefetched_sources"])
    else:
        assert OVERVIEW_URL not in urls
        assert "server_venture_stage_overview_fetch" not in document
    search.assert_awaited_once()
