"""A paid Firecrawl diagnostic must not masquerade as source content."""

import asyncio
import json

import pytest

from lab_arena import broker as br
from lab_arena import scoring_provider_compat as compat
from qualification.scoring import company_evidence_investigator as investigator
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT,
    DL_KEY,
    FakeLedgerStore,
    FakeTransport,
    make_broker,
)


URL = "https://example.com/about"
DIAGNOSTIC = "Provider account capacity details redacted."


def _route():
    return compat.route_for(
        kind="score", funding_source="miner_key",
        round_id="arena-2026-10-13", operation_id="scrapingdog.scrape",
        parameters={"url": URL},
    )


def _envelope(raw_html, *, source_url=URL, status=200):
    return {
        "job_id": "public-execute:firecrawl-diagnostic-test",
        "status": "completed",
        "result": {"data": {
            "rawHtml": raw_html,
            "metadata": {"sourceURL": source_url, "url": URL, "statusCode": status},
        }},
        "billing": {"credits_charged": 0.02, "cost_usd": 0.002},
    }


@pytest.mark.parametrize("raw_html", ["", " \n ", DIAGNOSTIC, "  Provider  account\ncapacity details redacted.  "])
def test_successful_firecrawl_requires_source_content(raw_html):
    with pytest.raises(compat.CompatibilityResponseError, match="firecrawl_source_content_unavailable"):
        compat.adapt_response_with_trusted_url(
            _route(), status=200, headers={},
            body=json.dumps(_envelope(raw_html)).encode(),
        )


@pytest.mark.parametrize("raw_html", [
    "Short but valid page text.",
    "<article>Provider account capacity details redacted. This article explains the issue.</article>",
])
def test_valid_short_or_mentioning_page_survives(raw_html):
    status, _headers, body, response_url = compat.adapt_response_with_trusted_url(
        _route(), status=200, headers={},
        body=json.dumps(_envelope(raw_html)).encode(),
    )
    assert (status, body, response_url) == (200, raw_html.encode(), URL)


def test_wrong_url_and_missing_status_still_fail_before_content_check():
    with pytest.raises(compat.CompatibilityResponseError, match="firecrawl_source_url_mismatch"):
        compat.adapt_response_with_trusted_url(
            _route(), status=200, headers={},
            body=json.dumps(_envelope(DIAGNOSTIC, source_url="https://other.example/")).encode(),
        )
    missing_status = _envelope(DIAGNOSTIC)
    del missing_status["result"]["data"]["metadata"]["statusCode"]
    with pytest.raises(compat.CompatibilityResponseError, match="missing_firecrawl_target_status"):
        compat.adapt_response_with_trusted_url(
            _route(), status=200, headers={}, body=json.dumps(missing_status).encode(),
        )


def _score_context():
    return br.RunContext(**{**CONTEXT.__dict__, "kind": "score", "round_id": "arena-2026-10-13"})


def test_paid_diagnostic_settles_exact_cost_once_and_triggers_fetch_failure(monkeypatch):
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    broker, _store, transport = make_broker(
        store=store, transport=FakeTransport([(200, _envelope(DIAGNOSTIC))]),
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(
        _score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": URL}, action_sequence=0, timeout_ms=30_000,
    )
    assert result.status == 502 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 2_000
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    saved = store.calls[result.call["call_identity"]]
    assert saved["kind"] == "settlement" and saved["actual"] == 2_000
    assert saved["terminal"]["call_succeeded"] is False
    assert saved["terminal"]["provider_cost"]["units"] == "0.02"
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1

    replay = broker.execute(
        _score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": URL}, action_sequence=0, timeout_ms=30_000,
    )
    assert replay.status == 502 and replay.body == result.body
    assert replay.call["call_identity"] == result.call["call_identity"]
    assert store.calls[result.call["call_identity"]]["actual"] == 2_000
    assert len(transport.sent) == 1 and store.log.count("settle") == 1

    async def fetch(_session, _url, **_kwargs):
        return result.status, URL, result.body.decode()

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-key")
    monkeypatch.setattr(investigator, "_fetch_bounded_html", fetch)
    page = asyncio.run(investigator._fetch_page(object(), URL))
    assert page == {"ok": False, "error": "http_502"}


def test_diagnostic_with_delayed_billing_keeps_full_reservation():
    payload = _envelope(DIAGNOSTIC)
    payload["billing"] = {"pricing_status": "pending"}
    transport = FakeTransport([(200, payload), (200, {"entries": "invalid"})])
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    broker, _store, _transport = make_broker(
        store=store, transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(
        _score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": URL}, action_sequence=0, timeout_ms=30_000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert result.call["reserved_microusd"] == 49_945_650
    assert "actual_microusd" not in result.call
    saved = store.calls[result.call["call_identity"]]
    assert saved["kind"] == "uncertain" and saved["amount"] == 49_945_650
    assert saved["uncertain_doc"]["failure_stage"] == "response_adaptation"
    assert saved["uncertain_doc"]["error_class"] == "CompatibilityResponseError"
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert len([request for request in transport.sent if request["method"] == "POST"]) == 1
