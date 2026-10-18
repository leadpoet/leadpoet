"""A missing frozen score route is not a provider or credential outage."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from lab_arena import runner, service
from qualification.scoring import intent_verification_three_stage as intent
from tests.lab_arena.deepline_worker_same_action_recovery_test import Api, worker
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeTransport, make_broker,
)


ROUND = "arena-2026-10-18"
MISSING_TOOL_CATALOG = frozen(row("vendor_company_search"))
POST_URL = "https://www.linkedin.com/posts/example_activity-7489978607814144000"


def score_context(catalog=MISSING_TOOL_CATALOG):
    return replace(CONTEXT, kind="score", round_id=ROUND,
                   deepline_catalog=catalog)


def routed_broker(*, funding_source="host", transport=None):
    options = ({
        "host_shadow_score_compat_round_id": ROUND,
        "host_shadow_score_compat_source_round_id": ROUND,
    } if funding_source == "host" else {})
    return make_broker(
        transport=transport or FakeTransport(),
        provider_funding_source_for=lambda _context, _provider: funding_source,
        **options,
    )


def firecrawl_catalog(*, max_url_length=None):
    url_schema = {"type": "string"}
    if max_url_length is not None:
        url_schema["maxLength"] = max_url_length
    schema = {"type": "object", "properties": {
        "url": url_schema,
        "formats": {"type": "array", "items": {"type": "string"}},
        "onlyMainContent": {"type": "boolean"},
        "maxAge": {"type": "integer"},
        "timeout": {"type": "integer"},
        "storeInCache": {"type": "boolean"},
    }, "required": ["url", "formats", "onlyMainContent", "maxAge",
                    "timeout", "storeInCache"]}
    return frozen(row("firecrawl_scrape", provider="firecrawl",
                      categories=["research"], inputSchema=schema))


@pytest.mark.parametrize(("operation", "parameters"), [
    ("scrapingdog.x_post", {"tweetId": "2039690176843170070"}),
    ("scrapingdog.linkedinjobs", {"job_id": "1234567890"}),
    ("scrapingdog.profile_post", {"id": "7489978607814144000"}),
])
@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
def test_missing_frozen_score_tool_is_truthful_and_never_dispatched(
    operation, parameters, funding_source,
):
    broker, store, transport = routed_broker(funding_source=funding_source)
    result = broker.execute(
        score_context(), operation_id=operation, parameters=parameters,
        action_sequence=1, timeout_ms=30_000,
    )
    assert result.status == 400
    assert json.loads(result.body) == {
        "error": {"code": "capability_unavailable"},
    }
    assert result.call["outcome"] == "not_dispatched"
    assert result.call["actual_microusd"] == 0
    assert result.call["funding_source"] == funding_source
    assert "call_identity" not in result.call
    assert store.calls == {} and store.log == [] and transport.sent == []


def test_direct_bad_miner_tool_stays_invalid_request():
    broker, store, transport = routed_broker(funding_source="miner_key")
    result = broker.execute(
        score_context(), operation_id="deepline.execute",
        parameters={"tool": "not_frozen_tool", "payload": {"query": "x"}},
        action_sequence=1, timeout_ms=30_000,
    )
    assert result.status == 400
    assert json.loads(result.body) == {"error": {"code": "invalid_request"}}
    assert store.calls == {} and transport.sent == []


@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
def test_present_frozen_score_tool_still_dispatches_and_adapts(funding_source):
    source = "https://example.com/about"
    transport = FakeTransport([(200, {
        "status": "completed", "billing": {"credits_charged": 0.02},
        "result": {"data": {"rawHtml": "<html>Company facts</html>",
                            "metadata": {"url": source, "sourceURL": source,
                                         "statusCode": 200}}},
    })])
    broker, store, _ = routed_broker(funding_source=funding_source,
                                      transport=transport)
    result = broker.execute(
        score_context(firecrawl_catalog()), operation_id="scrapingdog.scrape",
        parameters={"url": source}, action_sequence=1, timeout_ms=30_000,
    )
    assert result.status == 200
    assert result.call["outcome"] == "settled"
    assert len(store.calls) == 1 and len(transport.sent) == 1
    assert transport.sent[0]["method"] == "POST"


@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
def test_present_frozen_tool_provider_failure_is_not_masked(funding_source):
    transport = FakeTransport([(502, {"error": {"code": "upstream_failure"}})])
    broker, store, _ = routed_broker(funding_source=funding_source,
                                      transport=transport)
    result = broker.execute(
        score_context(firecrawl_catalog()), operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=1, timeout_ms=1_000,
    )
    assert result.status == 502
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["outcome"] == "uncertain"
    assert transport.sent[0]["method"] == "POST"
    assert len(store.calls) == 1


@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
def test_invalid_frozen_catalog_remains_broker_failure_without_dispatch(
    funding_source,
):
    broker, store, transport = routed_broker(funding_source=funding_source)
    result = broker.execute(
        score_context({"tools": []}), operation_id="scrapingdog.profile_post",
        parameters={"id": "7489978607814144000"},
        action_sequence=1, timeout_ms=30_000,
    )
    assert result.status == 503
    assert json.loads(result.body) == {"error": {"code": "broker_unavailable"}}
    assert store.calls == {} and transport.sent == []


@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
def test_present_tool_with_invalid_payload_keeps_existing_broker_error(
    funding_source,
):
    broker, store, transport = routed_broker(funding_source=funding_source)
    result = broker.execute(
        score_context(firecrawl_catalog(max_url_length=5)),
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=1, timeout_ms=30_000,
    )
    assert result.status == 503
    assert json.loads(result.body) == {"error": {"code": "broker_unavailable"}}
    assert store.calls == {} and transport.sent == []


def test_missing_capability_is_telemetry_refusal_not_provider_failure():
    fields = service._provider_telemetry_fields({
        "operation_id": "scrapingdog.profile_post",
        "error_code": "capability_unavailable",
        "outcome": "not_dispatched",
    }, 400, 1)
    assert fields["error_code"] == "capability_unavailable"
    assert fields["outcome"] == "refused"


def test_worker_preserves_missing_capability_as_400_without_provider_fault(
    tmp_path,
):
    broker, _, _ = routed_broker()
    parameters = {"id": "7489978607814144000"}
    result = broker.execute(
        score_context(), operation_id="scrapingdog.profile_post",
        parameters=parameters, action_sequence=0, timeout_ms=30_000,
    )
    api = Api(result.to_document())
    host = worker(tmp_path, api)
    error, returned = host._dispatch_once(
        "scrapingdog.profile_post", parameters, 30_000,
    )
    assert error is None and returned["status"] == 400
    assert len(api.frames) == 1
    assert runner._unresolved_operation_failures(host._state.calls, set()) == (0, 0)


@pytest.mark.parametrize("exa_ok", [True, False])
def test_specialized_post_unavailable_uses_existing_source_fallback(
    monkeypatch, exa_ok,
):
    async def specialized(_url):
        return {"ok": False, "stage": "linkedin_post_http_400",
                "content": "", "error": "capability_unavailable"}

    async def generic(_url):
        return {"ok": False, "stage": "anti_bot", "content": "",
                "error": "source unavailable"}

    async def exa(_url):
        return {"ok": exa_ok, "stage": "exa_scraped" if exa_ok else "exa_no_results",
                "content": "Post text grounded to the exact source" if exa_ok else "",
                "error": None if exa_ok else "no content"}

    monkeypatch.setattr(intent, "_scrape_linkedin_post", specialized)
    monkeypatch.setattr(intent, "_scrape_sd_hardened", generic)
    monkeypatch.setattr(intent, "_scrape_exa", exa)
    fetched = asyncio.run(intent._fetch_sd_then_exa([POST_URL]))
    assert fetched["statuses"][0]["source"] == "scrapingdog_linkedin_post_failed"
    assert fetched["results"] == ([{
        "url": POST_URL, "title": "",
        "text": "Post text grounded to the exact source",
        "source_publication_date": "", "meta": {},
    }] if exa_ok else [])
    assert fetched["statuses"][-1]["source"] == (
        "exa_fallback" if exa_ok else "none"
    )
    projected = intent._project_contents_for_prompt(fetched)
    assert len(projected["results"]) == int(exa_ok)
    assert not intent._confirmed_source_absence(fetched, POST_URL)
    if exa_ok:
        assert projected["results"][0]["url"] == POST_URL
        assert projected["results"][0]["text"] == "Post text grounded to the exact source"


@pytest.mark.asyncio
async def test_specialized_unavailable_can_still_complete_grounded_judgment(
    monkeypatch,
):
    body = ("Acme launched a new product on 2026-09-14. "
            "The product gives customers better reporting.")

    async def specialized(_url):
        return {"ok": False, "stage": "linkedin_post_http_400",
                "content": "", "error": "capability_unavailable"}

    async def generic(_url):
        return {"ok": False, "stage": "anti_bot", "content": ""}

    async def exa(_url):
        return {"ok": True, "stage": "exa_scraped", "content": body}

    answer = {
        "signal_status": "supported", "confidence": "high",
        "same_entity_check": "pass", "verification_mode": "source_grounded",
        "evidence_urls_used": [POST_URL],
        "claim_matches_miner_date": "supported",
        "supporting_quotes": ["Acme launched a new product on 2026-09-14."],
    }
    first = {**answer, "signal_status": "unable_to_verify"}
    model = AsyncMock(side_effect=[
        {"answer": {"signal_evaluations": [first]}, "model": "test-model", "usage": {}},
        {"answer": {"signal_evaluations": [answer]}, "model": "test-model", "usage": {}},
    ])
    monkeypatch.setattr(intent, "_scrape_linkedin_post", specialized)
    monkeypatch.setattr(intent, "_scrape_sd_hardened", generic)
    monkeypatch.setattr(intent, "_scrape_exa", exa)
    monkeypatch.setattr(intent, "_call_openrouter", model)

    result = await intent.verify_three_stage(
        object(), company_name="Acme",
        company_linkedin="https://www.linkedin.com/company/acme",
        company_website="https://acme.example", source_url=POST_URL,
        miner_claim="Acme launched a new product.",
        target_signal_text="Launched a new product in the past year.",
        miner_signal_date="2026-09-14", evidence_type="PRODUCT_LAUNCH",
        declared_source="linkedin", integrity_policy=True,
    )
    assert result["client_ready"] is True and result["decision"] == "approve"
    assert result["verified_source_context"] == [{
        "url": POST_URL, "text": body, "source_publication_date": "",
    }]
    assert model.await_count == 2
