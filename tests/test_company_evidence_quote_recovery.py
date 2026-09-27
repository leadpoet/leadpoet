"""Bounded recovery after an exact-quote repair fails."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator


def _core_usage(usage):
    return {key: usage[key] for key in ("reasoning_turns", "search_calls", "fetch_calls")}


def _finding(target: str, **overrides):
    finding = {
        "target": target,
        "status": "VERIFIED",
        "observed_value": "Privacy and Security" if target == "industry" else "Public",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "Privacy and Security" if target == "industry" else "",
        "observed_subindustry": (
            "Cloud security and identity protection" if target == "industry" else ""
        ),
        "activity_role": "supplier_operator" if target == "industry" else "unresolved",
        "evidence_url": "",
        "evidence_quote": "",
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "",
    }
    finding.update(overrides)
    return finding


def _response(turn: int, name: str, arguments: dict):
    return 200, {"choices": [{"message": {"tool_calls": [{
        "id": f"call-{turn}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}}]}


def _set_keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")


@pytest.mark.parametrize("source_proves_claim", [True, False])
def test_repeated_quote_failure_can_use_second_existing_search(
    monkeypatch, source_proves_claim,
):
    first_url = "https://www.crowdstrike.com/"
    recovered_url = "https://www.crowdstrike.com/about-us/"
    quote = "CrowdStrike supplies cybersecurity software that protects endpoints."
    requests = []
    searches = []

    async def fake_post(_session, _url, *, headers, payload):
        requests.append(payload)
        turn = len(requests)
        if turn in {1, 4}:
            return _response(turn, "search_web", {
                "query": "CrowdStrike cybersecurity software company",
            })
        if turn == 5:
            return _response(turn, "fetch_page", {"url": recovered_url})
        return _response(turn, "submit_findings", {"findings": [
            _finding(
                "industry", evidence_url=(first_url if turn < 5 else recovered_url),
                evidence_quote=quote,
            ),
        ]})

    async def fake_search(_session, query, *, key):
        searches.append(query)
        return {"results": [{"url": recovered_url}]}

    fetch = AsyncMock(return_value={
        "ok": True, "url": recovered_url, "final_url": recovered_url,
        "text": quote if source_proves_claim else "CrowdStrike careers and offices.",
    })
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "CrowdStrike", "website": first_url},
        targets=("industry",), requested_industry="Privacy and Security",
        requested_attribute="Sells cybersecurity software or services.",
        prior_observations={"submitted_source_urls": [first_url]},
        prefetched_pages={first_url: {
            "final_url": first_url, "text": "CrowdStrike. We stop breaches.",
        }},
        verified_homepage_identity={
            "normalized_name": "CrowdStrike",
            "registrable_dns_domain": "crowdstrike.com",
        },
    ))

    assert result["claims"]["industry"]["status"] == (
        "VERIFIED" if source_proves_claim else "UNPROVEN"
    )
    assert len(searches) == result["usage"]["search_calls"] == 2
    assert result["usage"]["fetch_calls"] == 1
    fetch.assert_awaited_once()
    assert requests[3]["tool_choice"] == {
        "type": "function", "function": {"name": "search_web"},
    }
    if source_proves_claim:
        assert result["usage"]["reasoning_turns"] == 6


def test_ciso_quote_repair_failure_forces_one_remaining_search_and_fetch(monkeypatch):
    company_url = "https://www.ciso.inc/company/"
    sec_url = "https://www.sec.gov/Archives/edgar/data/1777319/form10-k.htm"
    company_text = (
        "At CISO Global, we partner with our clients throughout their "
        "cybersecurity journey to achieve cyber resilience."
    )
    invalid_quote = (
        "CISO Global operates cybersecurity programs for clients."
    )
    sec_quote = (
        "CISO Global, Inc. (NASDAQ: CISO), a provider of cybersecurity "
        "software and managed security services."
    )
    requests = []
    searches = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn <= 2:
            return _response(turn, "submit_findings", {"findings": [
                _finding("stage", status="UNPROVEN"),
                _finding(
                    "industry",
                    evidence_url=company_url,
                    evidence_quote=invalid_quote,
                ),
            ]})
        if turn == 3:
            return _response(turn, "search_web", {
                "query": "CISO Global current Nasdaq listing cybersecurity software",
            })
        if turn == 4:
            return _response(turn, "fetch_page", {"url": sec_url})
        return _response(turn, "submit_findings", {"findings": [
            _finding("stage", evidence_url=sec_url, evidence_quote=sec_quote),
            _finding(
                "industry",
                evidence_url=company_url,
                evidence_quote=company_text,
            ),
        ]})

    async def fake_search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": [{"url": sec_url}]}

    fetch = AsyncMock(return_value={
        "ok": True,
        "url": sec_url,
        "final_url": sec_url,
        "text": sec_quote,
    })
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "CISO Global", "website": "https://ciso.inc/"},
        targets=("stage", "industry"),
        requested_stage="Public",
        requested_industry="Privacy and Security",
        requested_subindustry="Cloud security and identity protection",
        requested_attribute=(
            "Sells security software or services used to detect, prevent, or "
            "respond to privacy or cybersecurity risk."
        ),
        prior_observations={"submitted_source_urls": [company_url]},
        prefetched_pages={
            company_url: {"final_url": company_url, "text": company_text},
        },
        verified_homepage_identity={
            "normalized_name": "CISO Global",
            "registrable_dns_domain": "ciso.inc",
        },
    ))

    assert result["claims"]["stage"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["status"] == "VERIFIED", (
        result["claims"]["industry"], result["usage"]
    )
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 5, "search_calls": 2, "fetch_calls": 1,
    }
    assert searches == [
        (
            "CISO Global ciso.inc current public listing completed "
            "take-private acquisition delisting"
        ),
        "CISO Global current Nasdaq listing cybersecurity software"
    ]
    fetch.assert_awaited_once()
    assert requests[1]["tool_choice"] == "required"
    assert requests[2]["tool_choice"] == {
        "type": "function", "function": {"name": "search_web"},
    }
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["investigation_limits"]["prefetched_pages"] == 1
    assert document["investigation_limits"]["remaining_fetch_calls"] == 3


def test_unibuddy_industry_only_quote_failure_uses_remaining_search(monkeypatch):
    homepage_url = "https://unibuddy.com/"
    grounded_attribute_quote = (
        "Our platform boosts enrollment by building trust and confidence "
        "through scalable peer-to-peer & community engagement."
    )
    exact_industry_quote = "Unibuddy\n" + grounded_attribute_quote
    absent_quote = (
        "Unibuddy operates a higher education enrollment and student "
        "engagement platform."
    )
    requests = []
    searches = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn <= 2:
            return _response(turn, "submit_findings", {"findings": [
                _finding(
                    "industry",
                    observed_value="Higher education student engagement platform",
                    observed_industry="Education",
                    observed_subindustry="Higher education services",
                    evidence_url=homepage_url,
                    evidence_quote=absent_quote,
                ),
            ]})
        if turn == 3:
            return _response(turn, "search_web", {
                "query": "Unibuddy higher education enrollment platform",
            })
        return _response(turn, "submit_findings", {"findings": [
            _finding(
                "industry",
                observed_value="Higher education student engagement platform",
                observed_industry="Education",
                observed_subindustry="Higher education services",
                evidence_url=homepage_url,
                evidence_quote=exact_industry_quote,
            ),
        ]})

    async def fake_search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": [{"url": homepage_url}]}

    fetch = AsyncMock()
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Unibuddy", "website": homepage_url},
        targets=("industry",),
        requested_industry="Education",
        requested_subindustry="Higher education services",
        requested_attribute=(
            "Operates a software or services platform used by education "
            "providers to manage enrollment, learning delivery, student "
            "communication, or campus operations."
        ),
        prior_observations={
            "observed_company_name": "Unibuddy",
            "observed_company_website": homepage_url,
            "submitted_source_urls": [homepage_url],
            "required_attribute_evidence_url": homepage_url,
            "required_attribute_evidence_quote": grounded_attribute_quote,
        },
        prefetched_pages={
            homepage_url: {
                "final_url": homepage_url,
                "text": exact_industry_quote,
            },
        },
        verified_homepage_identity={
            "normalized_name": "Unibuddy",
            "registrable_dns_domain": "unibuddy.com",
        },
    ))

    assert result["claims"] == {
        "industry": _finding(
            "industry",
            observed_value="Higher education student engagement platform",
            observed_industry="Education",
            observed_subindustry="Higher education services",
            evidence_url=homepage_url,
            evidence_quote=exact_industry_quote,
        )
    }
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 4,
        "search_calls": 1,
        "fetch_calls": 0,
    }
    assert searches == ["Unibuddy higher education enrollment platform"]
    fetch.assert_not_awaited()
    assert requests[2]["tool_choice"] == {
        "type": "function", "function": {"name": "search_web"},
    }
    document = json.loads(requests[0]["messages"][1]["content"].split("\n", 1)[1])
    assert document["requested_targets"] == ["industry"]
    assert "requested_stage" not in document or not document["requested_stage"]
    assert document["prior_observations"][
        "required_attribute_evidence_quote"
    ] == grounded_attribute_quote


def test_exact_quote_repair_from_existing_page_does_not_search(monkeypatch):
    url = "https://acme.example/security"
    exact_quote = "Acme supplies managed cybersecurity services to enterprises."
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        quote = "Acme is a cybersecurity supplier." if len(requests) == 1 else exact_quote
        return _response(len(requests), "submit_findings", {"findings": [
            _finding("industry", evidence_url=url, evidence_quote=quote),
        ]})

    search = AsyncMock()
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",),
        requested_industry="Privacy and Security",
        prior_observations={"submitted_source_urls": [url]},
        prefetched_pages={url: {"final_url": url, "text": exact_quote}},
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
        },
    ))

    assert result["claims"]["industry"]["status"] == "VERIFIED", (
        result["claims"]["industry"], result["usage"]
    )
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 2, "search_calls": 0, "fetch_calls": 0,
    }
    search.assert_not_awaited()
    assert requests[1]["tool_choice"] == "required"


def test_repeated_quote_failure_can_search_with_full_prefetch_set(
    monkeypatch,
):
    urls = [f"https://acme.example/source-{index}" for index in range(3)]
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) <= 2:
            finding = _finding(
                "industry",
                evidence_url=urls[0],
                evidence_quote="Acme is a cybersecurity supplier.",
            )
            return _response(
                len(requests), "submit_findings", {"findings": [finding]}
            )
        if len(requests) == 3:
            return _response(
                len(requests), "search_web", {"query": "Acme cybersecurity"}
            )
        else:
            finding = _finding("industry", status="UNPROVEN")
        return _response(len(requests), "submit_findings", {"findings": [finding]})

    search = AsyncMock(return_value={"results": []})
    fetch = AsyncMock()
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",),
        requested_industry="Privacy and Security",
        prior_observations={"submitted_source_urls": urls},
        prefetched_pages={
            url: {"final_url": url, "text": "Acme security source page."}
            for url in urls
        },
    ))

    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 4, "search_calls": 1, "fetch_calls": 0,
    }
    search.assert_awaited_once()
    fetch.assert_not_awaited()
    assert requests[2]["tool_choice"] == {
        "type": "function", "function": {"name": "search_web"},
    }


def test_unproven_with_capacity_mechanically_forces_instructed_search(monkeypatch):
    url = "https://acme.example/security"
    quote = "Acme supplies managed cybersecurity services to enterprises."
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            return _response(turn, "submit_findings", {
                "findings": [_finding("industry", status="UNPROVEN")],
            })
        if turn == 2:
            return _response(turn, "search_web", {"query": "Acme security services"})
        if turn == 3:
            return _response(turn, "fetch_page", {"url": url})
        return _response(turn, "submit_findings", {"findings": [
            _finding("industry", evidence_url=url, evidence_quote=quote),
        ]})

    search = AsyncMock(return_value={"results": [{"url": url}]})
    fetch = AsyncMock(return_value={
        "ok": True, "url": url, "final_url": url, "text": quote,
    })
    _set_keys(monkeypatch)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_fetch_page", fetch)

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",),
        requested_industry="Privacy and Security",
        verified_homepage_identity={
            "normalized_name": "Acme",
            "registrable_dns_domain": "acme.example",
        },
    ))

    assert result["claims"]["industry"]["status"] == "VERIFIED", (
        result["claims"]["industry"], result["usage"]
    )
    assert _core_usage(result["usage"]) == {
        "reasoning_turns": 4, "search_calls": 1, "fetch_calls": 1,
    }
    assert requests[1]["tool_choice"] == {
        "type": "function", "function": {"name": "search_web"},
    }
