"""A typed, target-bound Firecrawl site refusal uses the existing fallback."""

import asyncio
from dataclasses import replace
import json
from urllib.parse import urlsplit

import pytest

from lab_arena import broker as br, shim
from qualification.scoring import company_evidence_investigator as investigator
from tests.lab_arena.firecrawl_unsupported_source_test import routed_call
from tests.lab_arena.typed_investigator_fallback_test import _investigate
from tests.test_company_evidence_investigator import _finding


CAPTURED_URL = "https://forbusiness.snapchat.com/advertising/measurement"


def blocked_site_body(url=CAPTURED_URL, **changes):
    # The captured native classification and exact message shape. Only the
    # request ID is rebound to the existing controlled billing fixture.
    message = (
        f"Firecrawl does not support scraping {urlsplit(url).hostname}. "
        "The site is blocked by Firecrawl, not a credential problem."
    )
    return {
        "code": "UPSTREAM_BAD_INPUT", "error": message,
        "detail": message, "message": message,
        "provider": "firecrawl", "operation": "firecrawl_scrape",
        "credential_owner": "deepline_managed", "credential_source": "env",
        "failure_origin": "caller", "error_category": "validation",
        "upstream_status": 403, "request_id": "unsupported-source-request",
        "tool_error": {
            "code": "UPSTREAM_BAD_INPUT", "provider": "firecrawl",
            "operation": "firecrawl_scrape", "statusCode": 422,
            "origin": "caller", "category": "validation", "retryable": False,
        },
        **changes,
    }


@pytest.mark.parametrize("url", [CAPTURED_URL, "https://another-public.example/product"])
@pytest.mark.parametrize("credits,amount", [(0, 0), (0.5, 50_000)])
def test_bound_site_refusal_preserves_settlement_and_replay(url, credits, amount):
    result, broker, store, transport, context, arguments = routed_call(
        url=url, body=blocked_site_body(url), credits=credits,
    )
    assert result.status == 403
    assert result.body == b'{"error":{"code":"provider_request_refused"}}'
    assert result.call["provider_status"] == 422
    assert result.call["error_code"] == "provider_request_refused"
    assert result.call["actual_microusd"] == amount
    assert result.call["outcome"] == "settled"
    assert store.log == ["reserve", "dispatch", "settle"]
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert "account_failure_evidence" not in terminal
    assert "blocked by Firecrawl" not in repr(terminal)
    sent = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == result.status and replay.body == result.body
    assert replay.call["idempotent"] is True
    assert replay.call["actual_microusd"] == amount
    assert len(transport.sent) == sent


def test_captured_refusal_reaches_existing_exact_url_investigator_fallback(monkeypatch):
    response, *_ = routed_call(url=CAPTURED_URL, body=blocked_site_body())
    # The normal shim attaches the broker call identity to the HTTP response.
    response = replace(response, headers={
        **response.headers, shim.CALL_IDENTITY_HEADER: response.call["call_identity"],
    })
    result, calls = _investigate(monkeypatch, response, source_url=CAPTURED_URL)
    assert calls == {"fetch": 1, "exa": 1, "model": 2}
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert [item["ok"] for item in result["usage"]["fetch_outcomes"]] == [False, True]
    assert result["usage"]["fetch_calls"] == 2


def test_recovered_refusal_preserves_earlier_fetched_evidence_and_verified_finding(monkeypatch):
    good_url = "https://acme.example/platform"
    quote = "Acme provides cloud software for IT teams."
    response, *_ = routed_call(url=CAPTURED_URL, body=blocked_site_body())
    calls = {"model": 0, "exa": 0}

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def get(self, request_url, **kwargs):
            url = kwargs["params"]["url"]
            assert url in (good_url, CAPTURED_URL)
            return shim._AiohttpResponse(
                request_url=request_url,
                response_url=good_url if url == good_url else "",
                status=200 if url == good_url else response.status,
                headers=(
                    {"content-type": "text/html"} if url == good_url else
                    {**response.headers, shim.CALL_IDENTITY_HEADER: response.call["call_identity"]}
                ),
                body=quote.encode() if url == good_url else response.body,
            )

    async def post(_session, url, *, headers, payload):
        if url == "https://api.exa.ai/contents":
            calls["exa"] += 1
            assert payload["ids"] == [CAPTURED_URL]
            return 200, {"results": [{"url": CAPTURED_URL, "text": "Public product information."}]}
        calls["model"] += 1
        if calls["model"] > 1:
            assert quote in json.dumps(payload)
        if calls["model"] < 3:
            name = "fetch_page"
            arguments = {"url": good_url if calls["model"] == 1 else CAPTURED_URL}
        else:
            name = "submit_findings"
            arguments = {"findings": [_finding(
                "industry", observed_industry="Software",
                observed_subindustry="Cloud software", activity_role="supplier_operator",
                evidence_url=good_url, evidence_quote=quote,
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(calls["model"]), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}}]}

    async def search(*_, **__):
        return {"results": []}

    for name in ("OPENROUTER_API_KEY", "EXA_API_KEY", "SCRAPINGDOG_API_KEY"):
        monkeypatch.setenv(name, "offline-test-key")
    monkeypatch.setattr("socket.socket.connect", lambda *_: pytest.fail("network forbidden"))
    monkeypatch.setattr(investigator.aiohttp, "ClientSession", lambda **_: Session())
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry",), requested_industry="Software",
        verified_homepage_identity={
            "normalized_name": "Acme", "registrable_dns_domain": "acme.example",
        },
    ))
    assert calls == {"model": 3, "exa": 1}
    assert result["_completed_submit"] is True and result["failure_reason"] == ""
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert result["claims"]["industry"]["evidence_quote"] == quote
    pages = result[investigator.PRIVATE_FETCHED_PAGES_KEY]
    assert pages[good_url]["text"] == quote
    assert pages[CAPTURED_URL]["text"] == "Public product information."
    assert [item["ok"] for item in result["usage"]["fetch_outcomes"]] == [True, False, True]


@pytest.mark.parametrize("status", [200, 401, 402, 403, 429, 500, 502])
def test_other_statuses_never_become_site_refusals(status):
    assert not br._provider_request_refused(
        "deepline", {"tool": "firecrawl_scrape", "payload": {"url": CAPTURED_URL}},
        br.ProviderResponse(status, {}, json.dumps(blocked_site_body()).encode()),
    )


@pytest.mark.parametrize("provider,tool", [
    ("openrouter", "firecrawl_scrape"), ("deepline", "generic_http_request"),
])
def test_other_provider_or_tool_cannot_borrow_firecrawl_refusal(provider, tool):
    assert not br._provider_request_refused(
        provider, {"tool": tool, "payload": {"url": CAPTURED_URL}},
        br.ProviderResponse(422, {}, json.dumps(blocked_site_body()).encode()),
    )


@pytest.mark.parametrize("url", [
    "https://different.example/page", "http://forbusiness.snapchat.com/",
    "https://user@forbusiness.snapchat.com/", "https://forbusiness.snapchat.com:bad/",
    "https://127.0.0.1/", "https://localhost/", "https://metadata.internal/",
    "https://169.254.169.254/", "https://[::1]/", "https://2130706433/",
])
def test_unbound_or_private_target_does_not_become_site_refusal(url):
    body = (
        blocked_site_body()
        if "snapchat" in url or url == "https://different.example/page"
        else blocked_site_body(url)
    )
    assert not br._provider_request_refused(
        "deepline", {"tool": "firecrawl_scrape", "payload": {"url": url}},
        br.ProviderResponse(422, {}, json.dumps(body).encode()),
    )


@pytest.mark.parametrize("changes", [
    {"code": "INVALID_FORMATS"}, {"error": "Invalid formats"},
    {"detail": "Invalid formats"}, {"message": None},
    {"provider": "other"}, {"operation": "other"},
    {"failure_origin": "provider"}, {"error_category": "provider_auth"},
    {"credential_owner": "workspace"}, {"credential_source": "workspace"},
    {"upstream_status": 401}, {"tool_error": {}},
    {"tool_error": {"code": "UPSTREAM_BAD_INPUT", "retryable": True}},
])
def test_unproved_validation_or_account_failure_does_not_become_refusal(changes):
    assert not br._provider_request_refused(
        "deepline", {"tool": "firecrawl_scrape", "payload": {"url": CAPTURED_URL}},
        br.ProviderResponse(422, {}, json.dumps(blocked_site_body(**changes)).encode()),
    )


@pytest.mark.parametrize("body", [[], None, "bad", {"code": "UPSTREAM_BAD_INPUT"}])
def test_malformed_or_untyped_native_body_is_not_a_refusal(body):
    assert not br._provider_request_refused(
        "deepline", {"tool": "firecrawl_scrape", "payload": {"url": CAPTURED_URL}},
        br.ProviderResponse(422, {}, json.dumps(body).encode()),
    )


def test_invalid_native_json_is_not_a_refusal():
    assert not br._provider_request_refused(
        "deepline", {"tool": "firecrawl_scrape", "payload": {"url": CAPTURED_URL}},
        br.ProviderResponse(422, {}, b"{"),
    )
