from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.linkedin_company_size import (
    MALFORMED_RESPONSE_FAILURE_REASON,
)


URL = "https://acme.example/platform"
QUOTE = "Acme supplies enrollment software to universities worldwide."


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def _finding(
    *,
    status: str = "VERIFIED",
    quote: str = QUOTE,
    url: str = URL,
) -> dict[str, object]:
    return {
        "target": "industry",
        "status": status,
        "observed_value": (
            "University enrollment software" if status != "UNPROVEN" else None
        ),
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "Education" if status != "UNPROVEN" else "",
        "observed_subindustry": (
            "Higher education services" if status != "UNPROVEN" else ""
        ),
        "activity_role": (
            "supplier_operator" if status != "UNPROVEN" else "unresolved"
        ),
        "evidence_url": url if status != "UNPROVEN" else "",
        "evidence_quote": quote if status != "UNPROVEN" else "",
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "The first-party page describes Acme's enrollment software.",
    }


def _response(name: str, arguments: dict[str, object] | str, *, call: int = 1):
    raw_arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {"tool_calls": [{
                "id": f"call-{call}",
                "type": "function",
                "function": {"name": name, "arguments": raw_arguments},
            }]},
        }],
    }


def _run_industry(monkeypatch, post_json, clock: _Clock, **kwargs):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", post_json)
    monkeypatch.setattr(investigator, "time", clock)
    return asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",),
        requested_industry="Education",
        requested_subindustry="Higher education services",
        requested_product_service="Enrollment software for universities",
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={
            "normalized_name": "acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
        **kwargs,
    ))


@pytest.mark.parametrize("late_tool", ["search_web", "fetch_page"])
def test_late_tool_is_not_dispatched_and_loaded_evidence_is_judged(
    monkeypatch, late_tool,
):
    clock = _Clock()
    requests = []
    provider_calls = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            clock.now = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            )
            arguments = (
                {"query": "Acme enrollment software"}
                if late_tool == "search_web"
                else {"url": "https://acme.example/other"}
            )
            return 200, _response(late_tool, arguments)
        return 200, _response(
            "submit_findings", {"findings": [_finding()]}, call=2
        )

    async def fail_search(*_args, **_kwargs):
        provider_calls.append("search")
        raise AssertionError("late search must not be dispatched")

    async def fail_fetch(*_args, **_kwargs):
        provider_calls.append("fetch")
        raise AssertionError("late fetch must not be dispatched")

    monkeypatch.setattr(investigator, "_search_web", fail_search)
    monkeypatch.setattr(investigator, "_fetch_page", fail_fetch)
    result = _run_industry(monkeypatch, fake_post, clock)

    assert provider_calls == []
    assert len(requests) == 2
    assert requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }
    feedback = json.loads(requests[1]["messages"][-1]["content"])
    assert feedback["error"] == "judgment_time_reserved"
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["reasoning_turns"] == 2
    assert result["usage"]["search_calls"] == 0
    assert result["usage"]["fetch_calls"] == 0


def test_freshly_fetched_evidence_is_judged_after_late_search_is_withheld(
    monkeypatch,
):
    fresh_url = "https://acme.example/fresh-evidence"
    clock = _Clock()
    requests = []
    fetches = []
    searches = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            return 200, _response("fetch_page", {"url": fresh_url}, call=1)
        if len(requests) == 2:
            clock.now = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            )
            return 200, _response(
                "search_web", {"query": "Acme enrollment evidence"}, call=2
            )
        return 200, _response(
            "submit_findings",
            {"findings": [_finding(url=fresh_url)]},
            call=3,
        )

    async def fake_fetch(_session, url, *, stealth_mode=False):
        del stealth_mode
        fetches.append(url)
        return {"ok": True, "url": url, "final_url": url, "text": QUOTE}

    async def fail_search(*_args, **_kwargs):
        searches.append(True)
        raise AssertionError("late search must not be dispatched")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", fail_search)
    monkeypatch.setattr(investigator, "time", clock)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",),
        requested_industry="Education",
        requested_subindustry="Higher education services",
        requested_product_service="Enrollment software for universities",
        verified_homepage_identity={
            "normalized_name": "acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
    ))

    assert fetches == [fresh_url]
    assert searches == []
    assert len(requests) == 3
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["evidence_url"] == fresh_url
    assert result["usage"]["fetch_calls"] == 1
    assert result["usage"]["search_calls"] == 0


def test_reserved_judgment_cannot_turn_missing_evidence_positive(monkeypatch):
    clock = _Clock()
    requests = []
    fabricated_quote = "Acme sells a product that the loaded page never states."

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            clock.now = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            )
            return 200, _response(
                "fetch_page", {"url": "https://acme.example/missing"}
            )
        if len(requests) == 2:
            return 200, _response(
                "submit_findings",
                {"findings": [_finding(quote=fabricated_quote)]},
                call=2,
            )
        return 200, _response(
            "submit_findings",
            {"findings": [_finding(status="UNPROVEN")]},
            call=3,
        )

    async def fail_fetch(*_args, **_kwargs):
        raise AssertionError("late fetch must not be dispatched")

    monkeypatch.setattr(investigator, "_fetch_page", fail_fetch)
    result = _run_industry(monkeypatch, fake_post, clock)

    assert len(requests) == 3
    correction = json.loads(requests[2]["messages"][-1]["content"])
    assert correction["error"] == "deterministic_evidence_validation_failed"
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["claims"]["industry"]["evidence_quote"] == ""


def test_mandatory_current_stage_discovery_precedes_reserved_submit(monkeypatch):
    clock = _Clock()
    searches = []
    requests = []

    async def fake_search(_session, query, *, key):
        del key
        searches.append(query)
        clock.now = (
            investigator.ADMISSION_DEADLINE_SECONDS
            - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
        )
        return {"results": [], "notice": "discovery_only_not_evidence"}

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        finding = _finding(status="UNPROVEN")
        finding["target"] = "stage"
        return 200, _response("submit_findings", {"findings": [finding]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "time", clock)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
    ))

    assert len(searches) == 1
    assert len(requests) == 1
    assert requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["search_calls"] == 1


def test_reserve_overrides_pending_stage_recovery_without_marking_search_success(
    monkeypatch,
):
    clock = _Clock()
    searches = []
    requests = []

    async def unavailable_search(_session, query, *, key):
        del key
        searches.append(query)
        raise RuntimeError("search unavailable")

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            clock.now = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            )
            finding = _finding(quote="A quote absent from the loaded page.")
            finding.update(target="stage", observed_value="Private Equity")
            return 200, _response(
                "submit_findings", {"findings": [finding]}, call=1
            )
        finding = _finding(status="UNPROVEN")
        finding["target"] = "stage"
        return 200, _response(
            "submit_findings", {"findings": [finding]}, call=2
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_search_web", unavailable_search)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "time", clock)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Private Equity",
        verified_homepage_identity={
            "normalized_name": "acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))

    # The first search is the mandatory current-stage discovery. The rejected
    # quote schedules a recovery search, but the reserve replaces that pending
    # action with submit_findings and never treats the skipped search as done.
    assert len(searches) == 1
    assert len(requests) == 2
    assert requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["search_calls"] == 1


def test_reserved_submit_is_enforced_when_model_ignores_tool_choice(monkeypatch):
    calls = 0

    def monotonic():
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else (
            investigator.ADMISSION_DEADLINE_SECONDS
            - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
        )

    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, _response("search_web", {"query": "ignored constraint"})

    result = _run_industry(
        monkeypatch,
        fake_post,
        SimpleNamespace(monotonic=monotonic),
    )

    assert requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }
    assert result["claims"] == {}
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON


def test_truncated_reserved_submit_retry_remains_one_ordinary_turn(monkeypatch):
    clock = _Clock()
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            clock.now = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            )
            response = _response("submit_findings", '{"findings":[', call=1)
            response["choices"][0]["finish_reason"] = "length"
            return 200, response
        return 200, _response(
            "submit_findings", {"findings": [_finding()]}, call=2
        )

    result = _run_industry(
        monkeypatch,
        fake_post,
        clock,
        positive_semantic_review=True,
    )

    assert len(requests) == 2
    assert requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }
    assert requests[1]["reasoning"] == {"effort": "low"}
    assert result["claims"]["industry"]["status"] == "VERIFIED"


def test_reserve_does_not_activate_ninth_turn(
    monkeypatch,
):
    clock = _Clock()
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            return 200, _response(
                "fetch_page", {"url": "https://acme.example/other"}, call=1
            )
        if len(requests) == 2:
            clock.now = (
                investigator.ADMISSION_DEADLINE_SECONDS
                - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            )
            return 200, _response(
                "search_web", {"query": "Acme enrollment software"}, call=2
            )
        raise AssertionError("reserve must not activate the correction-only turn")

    async def fake_fetch(_session, url, *, stealth_mode=False):
        del stealth_mode
        return {
            "ok": True,
            "url": url,
            "final_url": url,
            "text": "No relevant evidence.",
        }

    async def fail_search(*_args, **_kwargs):
        raise AssertionError("late search must not be dispatched")

    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 2)
    monkeypatch.setattr(investigator, "_fetch_page", fake_fetch)
    monkeypatch.setattr(investigator, "_search_web", fail_search)
    result = _run_industry(monkeypatch, fake_post, clock)

    assert len(requests) == 2
    assert result["claims"] == {}
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON


@pytest.mark.parametrize("late_tool", ["search_web", "fetch_page"])
def test_inner_deadline_exit_retains_usage(monkeypatch, late_tool):
    clock = _Clock()
    provider_calls = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers, payload
        clock.now = investigator.ADMISSION_DEADLINE_SECONDS
        arguments = (
            {"query": "Acme enrollment software"}
            if late_tool == "search_web"
            else {"url": "https://acme.example/other"}
        )
        return 200, _response(late_tool, arguments)

    async def fail_provider(*_args, **_kwargs):
        provider_calls.append(True)
        raise AssertionError("provider tool must not be dispatched after deadline")

    monkeypatch.setattr(investigator, "_search_web", fail_provider)
    monkeypatch.setattr(investigator, "_fetch_page", fail_provider)
    result = _run_industry(monkeypatch, fake_post, clock)

    assert provider_calls == []
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["claims"]["industry"]["reason"] == (
        "investigation admission budget exhausted"
    )
    assert result["usage"] == {
        "reasoning_turns": 1,
        "search_calls": 0,
        "fetch_calls": 0,
        "prefetched_pages": 1,
        "total_loaded_pages": 1,
        "fetch_outcomes": [],
    }
