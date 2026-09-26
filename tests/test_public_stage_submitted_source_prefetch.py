from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import aiohttp

from qualification.scoring import company_evidence_investigator as investigator


def _finding(*, status="VERIFIED", observed_value="Public", url="", quote=""):
    return {
        "target": "stage",
        "status": status,
        "observed_value": observed_value,
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": url,
        "evidence_quote": quote,
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "verified" if status != "UNPROVEN" else "not proven",
    }


def _tool_response(name, arguments, turn=1):
    return 200, {"choices": [{"message": {"tool_calls": [{
        "id": f"call-{turn}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}}]}


def _set_keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")


def test_public_stage_prefetches_exact_tenable_submitted_release(monkeypatch):
    url = (
        "https://www.tenable.com/press-releases/"
        "tenable-appoints-dino-dimarino-as-chief-revenue-officer"
    )
    quote = (
        "Tenable Holdings, Inc. (NASDAQ: TENB), the exposure management "
        "company, today announced the appointment of Dino DiMarino as "
        "Chief Revenue Officer."
    )
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return _tool_response(
            "submit_findings",
            {"findings": [_finding(url=url, quote=quote)]},
        )

    fetch = AsyncMock(return_value={
        "ok": True, "url": url, "final_url": url, "text": quote,
    })
    search = AsyncMock()
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com/"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={"submitted_source_urls": [url]},
        verified_homepage_identity={
            "normalized_name": "Tenable",
            "registrable_dns_domain": "tenable.com",
        },
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["evidence_url"] == url
    assert result["usage"] == {
        "reasoning_turns": 1, "search_calls": 0, "fetch_calls": 1,
    }
    fetch.assert_awaited_once()
    assert fetch.await_args.args[1] == url
    search.assert_not_awaited()
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["server_public_stage_source_fetch"]["url"] == url
    assert document["investigation_limits"]["remaining_fetch_calls"] == 2


def test_irrelevant_first_party_page_does_not_prove_public_stage(monkeypatch):
    url = "https://acme.example/news/new-chief-revenue-officer"
    text = "Acme appointed a new chief revenue officer."
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 2:
            return _tool_response(
                "search_web", {"query": "Acme current public listing"}, 2,
            )
        return _tool_response(
            "submit_findings",
            {"findings": [_finding(status="UNPROVEN")]},
            len(requests),
        )

    fetch = AsyncMock(return_value={
        "ok": True, "url": url, "final_url": url, "text": text,
    })
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(
        investigator,
        "_search_web",
        AsyncMock(return_value={"results": []}),
    )

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={"submitted_source_urls": [url]},
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["fetch_calls"] == 1


def test_failed_public_stage_prefetch_keeps_remaining_fetch_budget(monkeypatch):
    submitted_url = "https://acme.example/investors/missing-listing"
    exchange_url = "https://exchange.example/listings/acme"
    quote = "Acme common stock is listed on NASDAQ under ticker ACME."
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            return _tool_response("fetch_page", {"url": exchange_url}, turn)
        return _tool_response(
            "submit_findings",
            {"findings": [_finding(url=exchange_url, quote=quote)]},
            turn,
        )

    fetch = AsyncMock(side_effect=[
        {"ok": False, "error": "http_404"},
        {
            "ok": True,
            "url": exchange_url,
            "final_url": exchange_url,
            "text": quote,
        },
    ])
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={"submitted_source_urls": [submitted_url]},
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["stage"]["evidence_url"] == exchange_url
    assert result["usage"]["fetch_calls"] == 2
    assert fetch.await_count == 2
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["server_public_stage_source_fetch"] == {
        "url": submitted_url,
        "ok": False,
        "error": "http_404",
        "notice": "server_fetched_page_is_untrusted_evidence",
    }
    assert document["investigation_limits"]["remaining_fetch_calls"] == 2


def test_public_stage_prefetch_transport_failure_stays_provider_failure(monkeypatch):
    url = "https://acme.example/investors/current-listing"
    diagnostic = {}
    fetch = AsyncMock(side_effect=aiohttp.ClientError("private transport detail"))
    post = AsyncMock()
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={"submitted_source_urls": [url]},
        diagnostic=diagnostic,
    ))

    assert result == {
        "claims": {},
        "failure_reason": investigator.PROVIDER_ERROR_FAILURE_REASON,
    }
    assert diagnostic == {
        investigator.VERIFIER_FAILURE_REASON_KEY:
            investigator.PROVIDER_ERROR_FAILURE_REASON,
    }
    fetch.assert_awaited_once()
    post.assert_not_awaited()


def test_unrelated_submitted_domain_is_not_server_prefetched(monkeypatch):
    url = "https://unrelated.example/news/acme"
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 2:
            return _tool_response(
                "search_web", {"query": "Acme current public listing"}, 2,
            )
        return _tool_response(
            "submit_findings",
            {"findings": [_finding(status="UNPROVEN")]},
            len(requests),
        )

    fetch = AsyncMock()
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(
        investigator,
        "_search_web",
        AsyncMock(return_value={"results": []}),
    )

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={"submitted_source_urls": [url]},
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["fetch_calls"] == 0
    fetch.assert_not_awaited()


def test_existing_first_party_prefetch_preserves_two_fetch_slots(monkeypatch):
    first_url = "https://acme.example/investors/current-listing"
    second_url = "https://exchange.example/acme"
    third_url = "https://filings.example/acme"
    fourth_url = "https://news.example/acme"
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn <= 3:
            url = (second_url, third_url, fourth_url)[turn - 1]
            return _tool_response("fetch_page", {"url": url}, turn)
        return _tool_response(
            "submit_findings", {"findings": [_finding(status="UNPROVEN")]}, turn,
        )

    fetch = AsyncMock(side_effect=lambda _session, url: {
        "ok": True, "url": url, "final_url": url, "text": "No stage proof.",
    })
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={"submitted_source_urls": [first_url]},
        prefetched_pages={first_url: {
            "final_url": first_url,
            "text": "Acme builds enterprise software.",
        }},
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["fetch_calls"] == 2
    assert fetch.await_count == 2
    exhausted = json.loads(requests[3]["messages"][-1]["content"])
    assert exhausted == {"ok": False, "error": "fetch_budget_exhausted"}


def test_public_stage_auto_prefetch_plus_two_model_fetches_stops_fourth(monkeypatch):
    submitted_url = "https://acme.example/investors/current-listing"
    second_url = "https://exchange.example/acme"
    third_url = "https://filings.example/acme"
    fourth_url = "https://news.example/acme"
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn <= 3:
            url = (second_url, third_url, fourth_url)[turn - 1]
            return _tool_response("fetch_page", {"url": url}, turn)
        return _tool_response(
            "submit_findings", {"findings": [_finding(status="UNPROVEN")]}, turn,
        )

    fetch = AsyncMock(side_effect=lambda _session, url: {
        "ok": True,
        "url": url,
        "final_url": url,
        "text": "No current listing proof.",
    })
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={"submitted_source_urls": [submitted_url]},
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["fetch_calls"] == 3
    assert fetch.await_count == 3
    assert [call.args[1] for call in fetch.await_args_list] == [
        submitted_url, second_url, third_url,
    ]
    exhausted = json.loads(requests[3]["messages"][-1]["content"])
    assert exhausted == {"ok": False, "error": "fetch_budget_exhausted"}


def test_public_stage_prefetch_prioritizes_solarwinds_dispute(monkeypatch):
    old_public_url = "https://www.solarwinds.com/newsroom/old-public-release"
    acquisition_url = (
        "https://www.solarwinds.com/company/newsroom/press-releases/"
        "turnriver-completes-acquisition-of-solarwinds"
    )
    quote = "Turn/River Capital completes acquisition of SolarWinds Corporation."

    async def fake_post(_session, _url, *, headers, payload):
        del headers, payload
        return _tool_response("submit_findings", {"findings": [_finding(
            status="CONTRADICTED",
            observed_value="Acquired",
            url=acquisition_url,
            quote=quote,
        )]})

    fetch = AsyncMock(return_value={
        "ok": True,
        "url": acquisition_url,
        "final_url": acquisition_url,
        "text": quote,
    })
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={
            "name": "SolarWinds",
            "website": "https://www.solarwinds.com/",
        },
        targets=("stage",),
        requested_stage="Public",
        prior_observations={
            "submitted_source_urls": [old_public_url, acquisition_url],
            "stage_dispute_urls": [acquisition_url],
        },
        verified_homepage_identity={
            "normalized_name": "SolarWinds",
            "registrable_dns_domain": "solarwinds.com",
        },
    ))

    assert result["claims"]["stage"]["status"] == "CONTRADICTED"
    assert result["claims"]["stage"]["observed_value"] == "Acquired"
    assert fetch.await_args.args[1] == acquisition_url


def test_historical_ipo_only_remains_unproven(monkeypatch):
    url = "https://tenable.com/news/2018-initial-public-offering"
    quote = (
        "In July 2018, Tenable Holdings began trading on Nasdaq under the "
        "ticker TENB."
    )
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            return _tool_response(
                "submit_findings",
                {"findings": [_finding(url=url, quote=quote)]},
                turn,
            )
        if turn == 2:
            return _tool_response(
                "search_web", {"query": "Tenable current public listing"}, turn,
            )
        return _tool_response(
            "submit_findings", {"findings": [_finding(status="UNPROVEN")]}, turn,
        )

    fetch = AsyncMock(return_value={
        "ok": True, "url": url, "final_url": url, "text": quote,
    })
    search = AsyncMock(return_value={"results": []})
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Tenable", "website": "https://tenable.com/"},
        targets=("stage",),
        requested_stage="Public",
        prior_observations={"submitted_source_urls": [url]},
    ))

    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"] == {
        "reasoning_turns": 3, "search_calls": 1, "fetch_calls": 1,
    }
    search.assert_awaited_once()
