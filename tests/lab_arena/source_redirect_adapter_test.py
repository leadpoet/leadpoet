"""Retain the observed M&G apex-to-www redirect without rewriting a source.

Observed receipt metadata: sourceURL=https://mandg.com/, statusCode=200,
url=https://www.mandg.com/, rawHtml=[REDACTED]. The HTML below is synthetic;
no captured page HTML, credentials, or response headers are retained here.
"""

import json
from dataclasses import replace

import pytest

from lab_arena import operations, scoring_provider_compat as compat


ORIGINAL = "https://mandg.com/"
OBSERVED_FINAL = "https://www.mandg.com/"
HTML = "<html><head><title>M&amp;G</title></head><body>Company page.</body></html>"


def route(*, fallback=False, response_url=None, source=ORIGINAL):
    return compat.route_for(
        kind="score", funding_source="miner_key", round_id="arena-2026-10-16",
        operation_id="scrapingdog.scrape",
        parameters={"url": source, "dynamic": True, "wait": 5000},
        timeout_ms=30_000, source_content_fallback=fallback,
        source_content_response_url=response_url,
    )


def firecrawl_document(*, source=ORIGINAL, final=OBSERVED_FINAL, status=200, html="[REDACTED]"):
    return {"status": "completed", "result": {"data": {
        "rawHtml": html,
        "metadata": {"sourceURL": source, "url": final, "statusCode": status},
    }}}


def generic_document(*, requested=OBSERVED_FINAL, final=OBSERVED_FINAL):
    return {
        "status": "completed", "job_id": "public-execute:source-redirect-test",
        "billing": {"credits_charged": 0, "cost_usd": 0},
        "toolResponse": {"view": "rawV2", "rawV2": {
            "provider": "generic_http", "operation": "generic_http_request",
            "method": "GET", "requested_url": requested, "final_url": final,
            "status_code": 200, "ok": True,
            "headers": {"content-type": "text/html; charset=utf-8"}, "data": HTML,
        }},
    }


def adapt(selected_route, document, *, headers=None):
    return compat.adapt_response_with_trusted_url(
        selected_route, status=200, headers=headers or {},
        body=json.dumps(document).encode(),
    )


@pytest.mark.parametrize("placeholder", [
    "[REDACTED]", "Provider account capacity details redacted.", "", " \n ",
])
def test_observed_redirect_repairs_same_source_without_changing_original_request(placeholder):
    primary = route()
    with pytest.raises(compat.CompatibilityResponseError) as captured:
        adapt(primary, firecrawl_document(html=placeholder))
    failure = captured.value
    assert str(failure) == "firecrawl_source_content_unavailable"
    assert failure.trusted_response_url == OBSERVED_FINAL

    fallback = route(fallback=True, response_url=failure.trusted_response_url)
    assert fallback.requested_parameters == primary.requested_parameters
    assert fallback.requested_parameters["url"] == ORIGINAL
    assert fallback.effective_parameters["payload"] == {
        "url": OBSERVED_FINAL, "method": "GET", "follow_redirects": True,
        "timeout_ms": 25_000,
    }
    assert adapt(fallback, generic_document()) == (
        200, {"content-type": "text/html; charset=utf-8"}, HTML.encode(), OBSERVED_FINAL,
    )


def test_missing_redirect_keeps_original_fallback_url_and_receipt_contract():
    fallback = route(fallback=True)
    assert fallback.effective_parameters["payload"]["url"] == ORIGINAL
    assert adapt(fallback, generic_document(requested=ORIGINAL))[2] == HTML.encode()
    with pytest.raises(compat.CompatibilityResponseError, match="generic_html_source_url_mismatch"):
        adapt(fallback, generic_document())


def test_only_actual_effective_url_may_identify_the_generic_request():
    fallback = route(fallback=True, response_url=OBSERVED_FINAL)
    for forgery in (ORIGINAL, "https://another.example/about/"):
        with pytest.raises(compat.CompatibilityResponseError, match="generic_html_source_url_mismatch"):
            adapt(fallback, generic_document(requested=forgery))
    # Durable replay may restore an older original-URL transport. Its receipt
    # still matches that exact restored payload, not the new redirect hint.
    replay = replace(fallback, effective_parameters={"tool": "generic_http_request", "payload": {
        **fallback.effective_parameters["payload"], "url": ORIGINAL,
    }})
    assert adapt(replay, generic_document(requested=ORIGINAL))[2] == HTML.encode()


@pytest.mark.parametrize("response_url", [
    "", "http://www.mandg.com/", "https://metadata.google.internal/",
    "https://private.local/", "https://user@www.mandg.com/",
    "https://www.mandg.com/?access_token=not-a-secret",
])
def test_explicit_untrusted_fallback_destination_fails_closed(response_url):
    with pytest.raises(operations.OperationRequestError):
        route(fallback=True, response_url=response_url)


@pytest.mark.parametrize("source", ["https://private.local/", "https://metadata.google.internal/"])
def test_public_destination_does_not_make_original_private_source_valid(source):
    with pytest.raises(operations.OperationRequestError):
        route(fallback=True, response_url=OBSERVED_FINAL, source=source)


@pytest.mark.parametrize("change,code", [
    ({"source": "https://another.example/"}, "firecrawl_source_url_mismatch"),
    ({"final": "https://private.local/"}, "invalid_firecrawl_final_url"),
    ({"final": "https://metadata.google.internal/"}, "invalid_firecrawl_final_url"),
    ({"final": "http://www.mandg.com/"}, "invalid_firecrawl_final_url"),
    ({"status": True}, "missing_firecrawl_target_status"),
    ({"status": "200"}, "missing_firecrawl_target_status"),
    ({"html": None}, "missing_firecrawl_raw_html"),
])
def test_bad_source_metadata_never_grants_redirect_authority(change, code):
    with pytest.raises(compat.CompatibilityResponseError, match=code) as captured:
        adapt(route(), firecrawl_document(**change))
    assert captured.value.trusted_response_url is None


def test_matching_private_source_metadata_never_grants_redirect_authority():
    private_source = "https://private.local/"
    with pytest.raises(compat.CompatibilityResponseError, match="invalid_firecrawl_source_url") as captured:
        adapt(route(source=private_source), firecrawl_document(source=private_source))
    assert captured.value.trusted_response_url is None


@pytest.mark.parametrize("status", [301, 401, 403, 404, 429, 500])
def test_non_success_target_placeholder_has_no_redirect_exception(status):
    result = adapt(route(), firecrawl_document(status=status))
    assert result[0] == status
    assert result[2] == b"[REDACTED]"


def test_provider_header_cannot_supply_or_replace_validated_redirect_authority():
    with pytest.raises(compat.CompatibilityResponseError) as captured:
        adapt(route(), firecrawl_document(), headers={
            operations.TRUSTED_RESPONSE_URL_HEADER: "https://foreign.example/",
        })
    assert captured.value.trusted_response_url == OBSERVED_FINAL
    assert compat.CompatibilityResponseError("missing_firecrawl_raw_html").trusted_response_url is None


def test_original_primary_and_healthy_content_ignore_fallback_hint():
    primary = route(response_url="https://foreign.example/")
    assert primary.effective_parameters["payload"]["url"] == ORIGINAL
    assert adapt(primary, firecrawl_document(html=HTML))[2:] == (HTML.encode(), OBSERVED_FINAL)


def test_existing_fragment_normalization_precedes_redirect_retention():
    with pytest.raises(compat.CompatibilityResponseError) as captured:
        adapt(route(), firecrawl_document(final=OBSERVED_FINAL + "#section"))
    assert captured.value.trusted_response_url == OBSERVED_FINAL
