from __future__ import annotations

import asyncio
import copy
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.linkedin_company_size import (
    PROVIDER_ERROR_FAILURE_REASON,
    VERIFIER_FAILURE_REASON_KEY,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _run_post(monkeypatch, responses, *, started=None):
    requests = []

    async def fake_post_json(_session, url, *, headers, payload):
        requests.append((url, copy.deepcopy(headers), copy.deepcopy(payload)))
        response = responses[len(requests) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    sleep = AsyncMock()
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator.asyncio, "sleep", sleep)
    result = asyncio.run(investigator._post_openrouter_json(
        object(),
        OPENROUTER_URL,
        headers={"Authorization": "Bearer test"},
        payload={"model": "openai/gpt-6-luna", "messages": [{"role": "user", "content": "review"}]},
        started=time.monotonic() if started is None else started,
    ))
    return result, requests, sleep


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_explicit_transient_status_retries_exact_payload_once(monkeypatch, status):
    result, requests, sleep = _run_post(
        monkeypatch,
        [(status, {"error": {"code": "provider_unavailable"}}), (200, {"ok": True})],
    )

    assert result == (200, {"ok": True})
    assert len(requests) == 2
    assert requests[1] == requests[0]
    sleep.assert_awaited_once_with(
        investigator.OPENROUTER_TRANSIENT_RETRY_DELAY_SECONDS
    )


def test_successful_initial_request_has_no_extra_call(monkeypatch):
    result, requests, sleep = _run_post(monkeypatch, [(200, {"ok": True})])

    assert result == (200, {"ok": True})
    assert len(requests) == 1
    sleep.assert_not_awaited()


def test_nontransient_status_has_no_retry(monkeypatch):
    result, requests, sleep = _run_post(
        monkeypatch,
        [(400, {"error": {"code": "invalid_request"}})],
    )

    assert result[0] == 400
    assert len(requests) == 1
    sleep.assert_not_awaited()


def test_second_transient_status_is_returned_without_a_third_call(monkeypatch):
    result, requests, sleep = _run_post(
        monkeypatch,
        [(502, {"error": {"code": "provider_unavailable"}})] * 2,
    )

    assert result[0] == 502
    assert len(requests) == 2
    sleep.assert_awaited_once()


def test_retry_delay_must_fit_before_admission_deadline(monkeypatch):
    monkeypatch.setattr(
        investigator,
        "time",
        SimpleNamespace(monotonic=lambda: 109.0),
    )
    result, requests, sleep = _run_post(
        monkeypatch,
        [(502, {"error": {"code": "provider_unavailable"}})],
        started=0.0,
    )

    assert result[0] == 502
    assert len(requests) == 1
    sleep.assert_not_awaited()


def test_retry_dispatch_is_rechecked_after_delay(monkeypatch):
    monotonic_values = iter((100.0, 111.0))
    monkeypatch.setattr(
        investigator,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )
    result, requests, sleep = _run_post(
        monkeypatch,
        [(502, {"error": {"code": "provider_unavailable"}})],
        started=0.0,
    )

    assert result[0] == 502
    assert len(requests) == 1
    sleep.assert_awaited_once()


def test_ambiguous_connection_failure_is_not_retried(monkeypatch):
    error = aiohttp.ClientConnectionError("response unavailable")
    requests = []

    async def fail_once(_session, url, *, headers, payload):
        requests.append((url, headers, payload))
        raise error

    sleep = AsyncMock()
    monkeypatch.setattr(investigator, "_post_json", fail_once)
    monkeypatch.setattr(investigator.asyncio, "sleep", sleep)

    with pytest.raises(aiohttp.ClientConnectionError) as caught:
        asyncio.run(investigator._post_openrouter_json(
            object(),
            OPENROUTER_URL,
            headers={"Authorization": "Bearer test"},
            payload={"model": "openai/gpt-6-luna", "messages": []},
            started=time.monotonic(),
        ))

    assert caught.value is error
    assert len(requests) == 1
    sleep.assert_not_awaited()


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _unproven_stage_finding() -> dict:
    return {
        "target": "stage",
        "status": "UNPROVEN",
        "observed_value": None,
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "",
        "observed_subindustry": "",
        "activity_role": "unresolved",
        "evidence_url": "",
        "evidence_quote": "",
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "No current stage proof was found.",
    }


def test_retry_preserves_tools_history_context_and_semantic_turn_count(monkeypatch):
    requests = []

    async def fake_post_json(_session, url, *, headers, payload):
        del headers
        assert url == OPENROUTER_URL
        requests.append(copy.deepcopy(payload))
        if len(requests) == 1:
            return 502, {"error": {"code": "provider_unavailable"}}
        if len(requests) == 2:
            return 200, {"choices": [{"message": {"tool_calls": [
                _tool_call("search", "search_web", {"query": "Acme current stage"})
            ]}}]}
        return 200, {"choices": [{"message": {"tool_calls": [
            _tool_call(
                "submit",
                "submit_findings",
                {"findings": [_unproven_stage_finding()]},
            )
        ]}}]}

    search = AsyncMock(return_value={
        "results": [],
        "notice": "discovery_only_not_evidence",
    })
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator.asyncio, "sleep", AsyncMock())

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
        positive_semantic_review=True,
    ))

    assert len(requests) == 3
    assert requests[1] == requests[0]
    assert requests[0]["tools"] == requests[1]["tools"]
    assert requests[0]["messages"] == requests[1]["messages"]
    assert requests[0]["tool_choice"] == requests[1]["tool_choice"] == "required"
    assert requests[0]["model"] == investigator.POSITIVE_SEMANTIC_REVIEW_MODEL
    assert requests[2]["messages"][:2] == requests[0]["messages"]
    assert [message["role"] for message in requests[2]["messages"][-2:]] == [
        "assistant", "tool",
    ]
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["reasoning_turns"] == 2
    assert result["usage"]["search_calls"] == investigator.MAX_SEARCH_CALLS
    assert search.await_count == investigator.MAX_SEARCH_CALLS


def test_exhausted_transient_retry_keeps_provider_error(monkeypatch):
    requests = []

    async def unavailable(_session, _url, *, headers, payload):
        del headers, payload
        requests.append(1)
        return 502, {"error": {"code": "provider_unavailable"}}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", unavailable)
    monkeypatch.setattr(
        investigator,
        "_search_web",
        AsyncMock(return_value={
            "results": [],
            "notice": "discovery_only_not_evidence",
        }),
    )
    monkeypatch.setattr(investigator.asyncio, "sleep", AsyncMock())
    diagnostic = {}

    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("stage",),
        requested_stage="Public",
        diagnostic=diagnostic,
    ))

    assert len(requests) == 2
    assert result == {"claims": {}, "failure_reason": PROVIDER_ERROR_FAILURE_REASON}
    assert diagnostic[VERIFIER_FAILURE_REASON_KEY] == PROVIDER_ERROR_FAILURE_REASON
