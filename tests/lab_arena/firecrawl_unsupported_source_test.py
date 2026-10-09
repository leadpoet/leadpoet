"""Unsupported LinkedIn sources retain billing and the existing proof gates."""

import asyncio
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, DL_KEY, FakeTransport, deepline_history, deepline_history_entry,
    make_broker,
)


UNSUPPORTED = (
    "Firecrawl does not support scraping LinkedIn URLs. Use a native "
    "HarvestAPI LinkedIn operation when it covers the requested shape."
)
POST_URL = "https://www.linkedin.com/posts/example_activity-1234567890123456789"


def unsupported_body(**changes):
    return {
        "code": "UPSTREAM_BAD_INPUT",
        "error": UNSUPPORTED,
        "detail": UNSUPPORTED,
        "message": UNSUPPORTED,
        "request_id": "unsupported-source-request",
        **changes,
    }


def routed_call(*, url=POST_URL, body=None, credits=0):
    entry = deepline_history_entry(
        "unsupported-source-request", "firecrawl_scrape", credits,
        provider="firecrawl",
    )
    entry.update(status="error", outcome="error")
    transport = FakeTransport([
        (422, unsupported_body() if body is None else body),
        (200, deepline_history(entry)),
    ])
    broker, store, _ = make_broker(
        transport=transport,
        credential_for=lambda *_: DL_KEY,
        funding_source_for=lambda _context: "miner_key",
    )
    context = replace(CONTEXT, kind="score", round_id="arena-2026-10-09")
    arguments = dict(
        operation_id="scrapingdog.scrape", parameters={"url": url},
        action_sequence=0, timeout_ms=30_000,
    )
    return broker.execute(context, **arguments), broker, store, transport, context, arguments


@pytest.mark.parametrize("url", [
    POST_URL,
    "https://www.linkedin.com/posts/another_activity-9876543210987654321",
    "https://linkedin.com/company/another-company/",
])
@pytest.mark.parametrize("credits,amount", [(0, 0), (0.5, 50_000)])
def test_unsupported_source_dispatch_settlement_and_replay(url, credits, amount):
    result, broker, store, transport, context, arguments = routed_call(
        url=url, credits=credits,
    )
    assert result.status == 403
    assert result.body == b'{"error":{"code":"provider_request_refused"}}'
    assert result.call["error_code"] == "provider_request_refused"
    assert result.call["provider_status"] == 422
    assert result.call["actual_microusd"] == amount
    assert result.call["outcome"] == "settled"
    assert result.call["adapter"] == "firecrawl_raw_html"
    assert store.log == ["reserve", "dispatch", "settle"]
    call = store.calls[result.call["call_identity"]]
    assert call["operation_id"] == "scrapingdog.scrape"
    assert call["provider"] == "deepline"
    assert call["terminal"]["call_succeeded"] is False
    assert "account_failure_evidence" not in call["terminal"]
    assert UNSUPPORTED not in repr(call)
    assert DL_KEY not in repr(call)
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == result.status and replay.body == result.body
    assert replay.call["call_identity"] == result.call["call_identity"]
    assert replay.call["actual_microusd"] == amount
    assert replay.call["error_code"] == "provider_request_refused"
    assert replay.call["idempotent"] is True
    assert len(transport.sent) == before
    assert [request["method"] for request in transport.sent] == ["POST", "GET"]


@pytest.mark.parametrize("provider,parameters,status,body", [
    ("openrouter", {"tool": "firecrawl_scrape", "payload": {"url": POST_URL}}, 422, unsupported_body()),
    ("deepline", {"tool": "generic_http_request", "payload": {"url": POST_URL}}, 422, unsupported_body()),
    ("deepline", {"tool": "firecrawl_scrape", "payload": {"url": "https://example.com/"}}, 422, unsupported_body()),
    ("deepline", {"tool": "firecrawl_scrape", "payload": {"url": "https://linkedin.com.example.com/posts/x"}}, 422, unsupported_body()),
    ("deepline", {"tool": "firecrawl_scrape", "payload": {"url": "https://user@linkedin.com/posts/x"}}, 422, unsupported_body()),
    ("deepline", {"tool": "firecrawl_scrape", "payload": {"url": "https://linkedin.com:bad/posts/x"}}, 422, unsupported_body()),
    ("deepline", {"tool": "firecrawl_scrape", "payload": {"url": POST_URL}}, 422, unsupported_body(code="OTHER_BAD_INPUT")),
    ("deepline", {"tool": "firecrawl_scrape", "payload": {"url": POST_URL}}, 422, unsupported_body(error="Invalid formats")),
    ("deepline", {"tool": "firecrawl_scrape", "payload": {"url": POST_URL}}, 422, unsupported_body(detail="Account unavailable")),
    ("deepline", {"tool": "firecrawl_scrape", "payload": {"url": POST_URL}}, 422, unsupported_body(message=None)),
    *[("deepline", {"tool": "firecrawl_scrape", "payload": {"url": POST_URL}}, status, unsupported_body())
      for status in (401, 402, 403, 429, 500, 502, 504)],
    ("deepline", {"tool": "firecrawl_scrape", "payload": {"url": POST_URL}}, 422, []),
])
def test_other_input_account_and_transport_errors_are_not_source_refusals(
    provider, parameters, status, body,
):
    response = br.ProviderResponse(status, {}, json.dumps(body).encode())
    assert not br._provider_request_refused(provider, parameters, response)


def test_other_422_keeps_original_rejection_and_settlement():
    body = unsupported_body(error="Invalid formats", detail="Invalid formats", message="Invalid formats")
    result, _broker, store, _transport, _context, _arguments = routed_call(body=body)
    assert result.status == 422
    assert result.call.get("error_code") != "provider_request_refused"
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert store.log == ["reserve", "dispatch", "settle"]


@pytest.mark.parametrize("completed,status,grounding,expected", [
    (True, "UNPROVEN", "unproven", True),
    (False, "UNPROVEN", "unproven", False),
    (True, "VERIFIED", "unproven", False),
    (True, "UNPROVEN", "invalid_evidence", False),
])
def test_normalized_response_keeps_completed_unproven_attribute_requirements(
    monkeypatch, completed, status, grounding, expected,
):
    result, *_ = routed_call()
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-key")

    async def bounded(_session, url, **_kwargs):
        return result.status, url, result.body.decode()

    monkeypatch.setattr(investigator, "_fetch_bounded_html", bounded)
    fetched = asyncio.run(investigator._fetch_page(object(), POST_URL))
    assert fetched == {"ok": False, "error": "provider_request_refused"}
    outcome = investigator._fetch_outcome(POST_URL, fetched)
    receipt = {
        "gate": "company_evidence_investigation",
        "completed_submitted_findings": completed,
        "failure_reason": "",
        "targets": ["required_attribute"],
        "claims": {"required_attribute": {"status": status}},
        "usage": {"fetch_outcomes": [{"ok": True}, {"ok": True}, outcome]},
    }
    verdict = {lead_scorer._REQUIRED_ATTRIBUTE_GROUNDING: {"status": grounding}}
    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        verdict, ("required_attribute",), investigation_receipt=receipt,
    ) is expected
    # A separate real transport fault still prevents company-local exhaustion.
    receipt["usage"]["fetch_outcomes"].append({"ok": False, "error_class": "http_error"})
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        verdict, ("required_attribute",), investigation_receipt=receipt,
    )
