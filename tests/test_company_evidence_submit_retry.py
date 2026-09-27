from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lab_arena import operations as arena_operations
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring.linkedin_company_size import (
    MALFORMED_RESPONSE_FAILURE_REASON,
    VERIFIER_FAILURE_REASON_KEY,
)


URL = "https://acme.example/platform"
QUOTE = "Acme supplies enrollment software to universities worldwide."


def _finding(**overrides):
    finding = {
        "target": "industry",
        "status": "VERIFIED",
        "observed_value": "University enrollment software",
        "observed_country": "",
        "observed_state": "",
        "observed_industry": "Education",
        "observed_subindustry": "Higher education services",
        "activity_role": "supplier_operator",
        "evidence_url": URL,
        "evidence_quote": QUOTE,
        "old_name": "",
        "new_name": "",
        "old_domain": "",
        "new_domain": "",
        "shared_linkedin_slug": "",
        "reason": "The first-party page describes Acme's enrollment software.",
    }
    finding.update(overrides)
    return finding


def _tool_response(
    name: str,
    arguments: str,
    *,
    finish_reason: str = "tool_calls",
    completion_tokens: int | None = None,
    call_id: str = "call-1",
):
    response = {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }]},
        }],
    }
    if completion_tokens is not None:
        response["usage"] = {"completion_tokens": completion_tokens}
    return response


def _run(monkeypatch, post_json, *, positive_semantic_review: bool = True):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", post_json)
    return asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",),
        requested_industry="Education",
        requested_subindustry="Higher education services",
        requested_product_service="Enrollment software for universities",
        positive_semantic_review=positive_semantic_review,
        prior_observations={"submitted_source_urls": [URL]},
        verified_homepage_identity={
            "normalized_name": "acme",
            "registrable_dns_domain": "acme.example",
            "linkedin_company_slug": "acme",
        },
        prefetched_pages={URL: {"final_url": URL, "text": QUOTE}},
    ))


@pytest.mark.parametrize("positive_semantic_review", [False, True])
def test_normal_reasoning_request_keeps_existing_provider_default(
    monkeypatch, positive_semantic_review,
):
    routed = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        normalized = arena_operations.validate_operation_request(
            "openrouter.chat", payload
        )
        outbound = arena_operations.build_outbound_request(
            "openrouter.chat", payload
        )
        routed.append((payload, normalized, json.loads(outbound.body)))
        arguments = json.dumps({"findings": [_finding()]})
        return 200, _tool_response("submit_findings", arguments)

    result = _run(
        monkeypatch,
        fake_post_json,
        positive_semantic_review=positive_semantic_review,
    )

    assert result["claims"]["industry"]["status"] == "VERIFIED"
    payload, normalized, outbound = routed[0]
    assert payload["max_tokens"] == normalized["max_tokens"] == 3000
    assert "reasoning" not in payload
    assert "reasoning" not in normalized
    assert "reasoning" not in outbound


@pytest.mark.parametrize(
    ("finish_reason", "completion_tokens"),
    [
        ("length", None),
        # Retained actions 80 and 108 reported tool_calls, but both consumed
        # the request's exact 3,000 completion-token allowance.
        ("tool_calls", investigator.REASONING_MAX_TOKENS),
    ],
)
def test_incomplete_submit_retries_once_inside_ordinary_turn_budget(
    monkeypatch, finish_reason, completion_tokens,
):
    requests = []
    outbound_requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        arena_operations.validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        outbound = arena_operations.build_outbound_request(
            "openrouter.chat", payload
        )
        outbound_requests.append(json.loads(outbound.body))
        if len(requests) == 1:
            return 200, _tool_response(
                "submit_findings",
                '{"findings":[{"evidence_quote":"Acme',
                finish_reason=finish_reason,
                completion_tokens=completion_tokens,
                call_id="call-incomplete",
            )
        return 200, _tool_response(
            "submit_findings",
            json.dumps({"findings": [_finding()]}),
            call_id="call-complete",
        )

    result = _run(monkeypatch, fake_post_json)

    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["usage"]["reasoning_turns"] == 2
    assert result["usage"]["search_calls"] == 0
    assert result["usage"]["fetch_calls"] == 0
    assert len(requests) == 2
    assert requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }
    assert "reasoning" not in requests[0]
    assert requests[1]["reasoning"] == {"effort": "low"}
    replayed_arguments = outbound_requests[1]["messages"][-2]["tool_calls"][0][
        "function"
    ]["arguments"]
    assert replayed_arguments.endswith('"Acme')
    retry_feedback = json.loads(requests[1]["messages"][-1]["content"])
    assert retry_feedback["error"] == "incomplete_submit_findings"
    assert "Do not search or fetch" in retry_feedback["instruction"]


def test_retry_still_applies_exact_evidence_validation(monkeypatch):
    requests = []
    fabricated_quote = "Acme supplies payroll software to hospitals worldwide."

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == 1:
            return 200, _tool_response(
                "submit_findings",
                '{"findings":[{"evidence_quote":"Acme',
                completion_tokens=investigator.REASONING_MAX_TOKENS,
            )
        if len(requests) == 2:
            arguments = json.dumps({"findings": [_finding(
                observed_value="Payroll software",
                observed_industry="Healthcare",
                observed_subindustry="Hospital operations",
                evidence_quote=fabricated_quote,
            )]})
            return 200, _tool_response("submit_findings", arguments)
        return 200, _tool_response(
            "submit_findings", json.dumps({"findings": [_finding()]})
        )

    result = _run(monkeypatch, fake_post_json)

    assert len(requests) == 3
    assert "reasoning" not in requests[0]
    assert requests[1]["reasoning"] == {"effort": "low"}
    assert "reasoning" not in requests[2]
    validation_feedback = json.loads(requests[2]["messages"][-1]["content"])
    assert validation_feedback["error"] == "deterministic_evidence_validation_failed"
    assert result["claims"]["industry"]["evidence_quote"] == QUOTE


@pytest.mark.parametrize("tool_name", ["search_web", "fetch_page", "unknown_tool"])
def test_output_exhaustion_does_not_retry_other_malformed_tools(
    monkeypatch, tool_name,
):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, _tool_response(tool_name, "{", finish_reason="length")

    result = _run(monkeypatch, fake_post_json)

    assert len(requests) == 1
    assert result["claims"] == {}
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON


def test_multiple_tool_calls_remain_immediately_malformed(monkeypatch):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        call = {
            "id": "call-incomplete",
            "type": "function",
            "function": {"name": "submit_findings", "arguments": "{"},
        }
        return 200, {
            "choices": [{
                "finish_reason": "length",
                "message": {"tool_calls": [call, call]},
            }],
        }

    result = _run(monkeypatch, fake_post_json)

    assert len(requests) == 1
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON


def test_incomplete_submit_does_not_use_ninth_or_second_retry(monkeypatch):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, _tool_response(
            "submit_findings", '{"findings":[', finish_reason="length"
        )

    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 2)
    diagnostic = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(investigator, "_post_json", fake_post_json)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",),
        diagnostic=diagnostic,
    ))

    assert len(requests) == 2
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
    assert diagnostic[VERIFIER_FAILURE_REASON_KEY] == MALFORMED_RESPONSE_FAILURE_REASON


def test_incomplete_submit_retry_requires_remaining_admission_time(monkeypatch):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, _tool_response(
            "submit_findings", '{"findings":[', finish_reason="length"
        )

    monotonic_values = iter((0.0, 0.0, investigator.ADMISSION_DEADLINE_SECONDS))
    monkeypatch.setattr(
        investigator,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )
    result = _run(monkeypatch, fake_post_json)

    assert len(requests) == 1
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON


def test_arbitrary_malformed_submit_is_not_retried(monkeypatch):
    requests = []

    async def fake_post_json(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        return 200, _tool_response("submit_findings", '{"findings":[')

    result = _run(monkeypatch, fake_post_json)

    assert len(requests) == 1
    assert result["failure_reason"] == MALFORMED_RESPONSE_FAILURE_REASON
