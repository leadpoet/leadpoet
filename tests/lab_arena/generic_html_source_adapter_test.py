"""A source fallback needs the complete authenticated public-GET receipt."""

import json
from dataclasses import replace

import pytest

from lab_arena import operations, scoring_provider_compat as compat


SOURCE = "https://example.com/about/"
FINAL = "https://www.example.com/about/"
HTML = (
    '<!doctype html><html><head><title>Example</title></head><body>'
    '<p>Example supplies workflow software.</p>'
    '<footer><a href="https://www.linkedin.com/company/example">Example</a></footer>'
    '</body></html>'
)


def route(**changes):
    arguments = dict(
        kind="score", funding_source="miner_key", round_id="arena-2026-10-16",
        operation_id="scrapingdog.scrape",
        parameters={"url": SOURCE, "dynamic": True, "wait": 5000},
        timeout_ms=60_000, source_content_fallback=True,
    )
    arguments.update(changes)
    return compat.route_for(**arguments)


def envelope():
    return {
        "job_id": "public-execute:example-source-recovery", "status": "completed",
        "billing": {"credits_charged": 0, "cost_usd": 0, "pricing_status": "final"},
        "toolResponse": {"view": "rawV2", "rawV2": {
            "provider": "generic_http", "operation": "generic_http_request",
            "method": "GET", "requested_url": SOURCE, "final_url": FINAL,
            "status_code": 200, "ok": True,
            "headers": {"content-type": "text/html; charset=utf-8"}, "data": HTML,
        }},
    }


def adapt(document, *, status=200):
    return compat.adapt_response_with_trusted_url(
        route(), status=status,
        headers={"x-lab-arena-response-url": "https://foreign.example/"},
        body=json.dumps(document).encode(),
    )


def test_primary_route_and_original_rendering_request_stay_unchanged():
    primary = route(source_content_fallback=False)
    assert primary.adapter == "firecrawl_raw_html"
    assert primary.effective_parameters == {"tool": "firecrawl_scrape", "payload": {
        "url": SOURCE, "formats": ["rawHtml"], "onlyMainContent": False,
        "maxAge": 0, "timeout": 55_000, "storeInCache": False,
    }}
    fallback = route()
    assert fallback.adapter == "generic_http_raw_html"
    assert fallback.requested_parameters == primary.requested_parameters
    assert fallback.effective_parameters == {"tool": "generic_http_request", "payload": {
        "url": SOURCE, "method": "GET", "follow_redirects": True, "timeout_ms": 55_000,
    }}
    assert fallback.adapter in compat.SOURCE_HTML_ADAPTERS
    assert primary.adapter in compat.SOURCE_HTML_ADAPTERS


@pytest.mark.parametrize("changes", [{"kind": "execute"}, {"funding_source": "host"}])
def test_internal_fallback_does_not_expand_scoring_route_eligibility(changes):
    assert route(**changes) is None


def test_internal_fallback_keeps_specialized_ats_route():
    ats = route(parameters={"url": "https://api.ashbyhq.com/posting-api/job-board/example"})
    assert ats.adapter == "generic_ats_json:ashby"
    assert ats.effective_parameters["payload"]["follow_redirects"] is False


def test_source_html_and_redirect_identity_survive_without_foreign_headers():
    document = envelope()
    document["toolResponse"]["rawV2"]["headers"].update({
        "set-cookie": "test-session=private", "x-lab-arena-response-url": "https://foreign.example/",
    })
    status, headers, body, final_url = adapt(document)
    assert (status, body, final_url) == (200, HTML.encode(), FINAL)
    assert headers == {"content-type": "text/html; charset=utf-8"}


@pytest.mark.parametrize("content_type", ["text/html", "TEXT/HTML; charset=windows-1252", "application/xhtml+xml"])
def test_supported_source_html_types_are_returned_as_utf8(content_type):
    document = envelope()
    document["toolResponse"]["rawV2"]["headers"] = {"Content-Type": content_type}
    document["toolResponse"]["rawV2"]["data"] = HTML + " café"
    status, headers, body, _ = adapt(document)
    assert status == 200 and body.endswith(" café".encode())
    assert headers["content-type"].endswith("charset=utf-8")


@pytest.mark.parametrize("field,value", [
    ("provider", "firecrawl"), ("operation", "another_request"), ("method", "POST"),
    ("requested_url", FINAL), ("requested_url", SOURCE + "#section"),
    ("final_url", ""), ("final_url", "http://example.com/about/"),
    ("final_url", "https://user@example.com/about/"),
    ("status_code", True), ("status_code", "200"), ("status_code", 99), ("status_code", 600),
    ("ok", "true"), ("ok", False), ("data", {}), ("data", None),
    ("headers", {}), ("headers", {"content-type": 1}),
    ("headers", {"content-type": "text/html", "Content-Type": "text/plain"}),
    ("headers", {"content-type": "application/pdf"}),
    ("headers", {"content-type": "application/json"}),
])
def test_unproved_or_wrong_source_response_fails_closed(field, value):
    document = envelope()
    document["toolResponse"]["rawV2"][field] = value
    with pytest.raises(compat.CompatibilityResponseError):
        adapt(document)


@pytest.mark.parametrize("field", [
    "provider", "operation", "method", "requested_url", "final_url", "status_code", "ok", "headers", "data",
])
def test_required_receipt_fields_cannot_be_missing(field):
    document = envelope()
    del document["toolResponse"]["rawV2"][field]
    with pytest.raises(compat.CompatibilityResponseError):
        adapt(document)


@pytest.mark.parametrize("body", ["", " \n ", "[REDACTED]", " [redacted] ",
    "Provider account capacity details redacted.", " Provider account\ncapacity details redacted. "])
def test_fallback_placeholder_cannot_be_source_evidence(body):
    document = envelope()
    document["toolResponse"]["rawV2"]["data"] = body
    with pytest.raises(compat.CompatibilityResponseError, match="generic_html_source_content_unavailable"):
        adapt(document)


@pytest.mark.parametrize("change", [
    lambda d: d.update(status="running"),
    lambda d: d.update(error={"code": "failed"}),
    lambda d: d.update(tool_error={"code": "failed"}),
    lambda d: d["toolResponse"].update(view="data"),
    lambda d: d.update(result={"data": HTML}, toolResponse=None),
])
def test_legacy_partial_or_failed_envelope_cannot_supply_provenance(change):
    document = envelope()
    change(document)
    with pytest.raises(compat.CompatibilityResponseError):
        adapt(document)


@pytest.mark.parametrize("status", [
    100, 199, 301, 302, 304, 307, 308, 400, 401, 403, 404, 408, 410,
    422, 429, 451, 500,
])
def test_failed_target_fetch_keeps_source_absence_or_retry_without_error_body(status):
    document = envelope()
    document["toolResponse"]["rawV2"].update(
        status_code=status, ok=False, headers={"content-type": "application/json"},
        data={"error": "target refused the request"},
    )
    returned_status, headers, body, final_url = adapt(document)
    outward_status = status if status in {404, 410} else 502
    assert (returned_status, body, final_url) == (outward_status, b"", FINAL)
    assert headers == {"content-type": "text/html; charset=utf-8"}


def test_provider_http_failure_is_not_a_target_status_receipt():
    with pytest.raises(compat.CompatibilityResponseError, match="deepline_http_error"):
        adapt(envelope(), status=503)


@pytest.mark.parametrize("url", [
    "https://metadata.google.internal/", "https://private.local/",
    "https://private.internal/", "https://private.localhost/",
    "https://private.test/", "https://private.invalid/",
    "https://METADATA.GOOGLE.INTERNAL/", "https://metadata.google.internal./",
    "https://127.0.0.1/", "https://169.254.169.254/", "https://[::1]/",
    "https://example.com/?access_token=not-a-secret",
])
def test_private_or_reserved_source_cannot_be_requested_or_accepted(url):
    with pytest.raises(operations.OperationRequestError):
        route(parameters={"url": url})
    document = envelope()
    document["toolResponse"]["rawV2"]["final_url"] = url
    with pytest.raises(compat.CompatibilityResponseError, match="invalid_generic_html_final_url"):
        adapt(document)
    # A matching receipt and route are insufficient when their source is not
    # public. Enforce the same check when consuming the provider response.
    document = envelope()
    document["toolResponse"]["rawV2"]["requested_url"] = url
    private_route = replace(route(), requested_parameters={"url": url})
    with pytest.raises(compat.CompatibilityResponseError, match="generic_html_source_url_mismatch"):
        compat.adapt_response_with_trusted_url(
            private_route, status=200, headers={}, body=json.dumps(document).encode(),
        )


@pytest.mark.parametrize("url", [
    "https://metadata.google.internal/", "https://private.local/",
])
def test_primary_firecrawl_contract_is_unchanged_for_existing_urls(url):
    primary = route(parameters={"url": url}, source_content_fallback=False)
    assert primary.adapter == "firecrawl_raw_html"
    assert primary.effective_parameters["payload"]["url"] == url
