"""An oversized script must not consume the source HTML text allowance.

The regression shape comes from a retained 4,927,652-byte Baringa page:
a complete 4,228,660-byte JSON-LD payload preceded every visible body link.
Only a generated payload is committed; the exact retained reply is opt-in.
"""
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from lab_arena import operations, scoring_provider_compat as compat
from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.verification_helpers import visible_html_link_labels, visible_html_text

SOURCE = "https://www.baringa.com/en/careers/experienced/"
POSTING = "https://job-boards.eu.greenhouse.io/baringa/jobs/4879336101"
BOARD_LINK = "https://job-boards.eu.greenhouse.io/baringa/jobs/4961910101"
LIMIT = operations.OPERATIONS["scrapingdog.scrape"].max_response_bytes
SMALL_JSONLD = '<script type="application/ld+json">{"@type":"Organization","name":"Baringa"}</script>'


def route():
    return compat.route_for(kind="score", funding_source="miner_key",
        round_id="arena-2026-10-26", operation_id="scrapingdog.scrape",
        parameters={"url": SOURCE, "dynamic": True, "wait": 5000})


def document(html, **metadata):
    return {"status": "completed", "result": {"data": {
        "rawHtml": html, "metadata": {"sourceURL": SOURCE, "url": SOURCE,
                                     "statusCode": 200, **metadata}}}}


def adapted(document):
    return compat.adapt_response_with_trusted_url(route(), status=200, headers={},
        body=json.dumps(document, ensure_ascii=False).encode())


def worker_html(document):
    status, headers, body, _final = adapted(document)
    return operations.sanitize_response("scrapingdog.scrape", status, headers, body)[2].decode()


def projected_bytes(html):
    return operations._project_oversized_html_scripts(
        html.encode(), operations.OPERATIONS["scrapingdog.scrape"].max_response_bytes)


def oversized_html(payload=None, *, suffix=None):
    if payload is None:
        payload = '{"jobs":"' + "x" * 4_228_649 + '"}'
    if suffix is None:
        suffix = f'<body><a href="{BOARD_LINK}">Apply for this role</a></body>'
    opening = '<ScRiPt type="application/ld+json" data-test="a > b">'
    closing = '</ScRiPt >'
    prefix = '<html><head>\n' + SMALL_JSONLD + '\n'
    middle = '</head>\n' + suffix + '</html>'
    return prefix + opening + payload + closing + middle, prefix + opening + closing + middle


def anchors(html, *, source=SOURCE, posting=POSTING):
    page = {"url": source, "text": "Careers", "meta": {
        "observed_ownership_links": intent._observed_ownership_links(html, source)}}
    return intent._ats_ownership_anchors([page], [posting], {"observed_domain": "baringa.com"})


def test_observed_large_jsonld_shape_recovers_body_before_existing_text_cap():
    original, expected = oversized_html()
    assert len(original.encode()) > LIMIT
    assert list(visible_html_link_labels(original.encode()[:LIMIT].decode())) == []
    status, headers, body, final = adapted(document(original))
    assert (status, final) == (200, SOURCE)
    assert headers == {"content-type": "text/html; charset=utf-8"}
    assert body == original.encode()  # Source adaptation preserves the raw reply.
    assert worker_html(document(original)) == expected
    assert list(visible_html_link_labels(expected)) == list(visible_html_link_labels(original))
    assert anchors(expected) and SMALL_JSONLD in expected
    assert worker_html(document(expected)) == expected  # No further source mutation.


@pytest.mark.parametrize("extra_bytes", [-1, 0, 1])
def test_payload_threshold_uses_operation_limit_in_utf8_bytes(extra_bytes):
    # More than LIMIT bytes, but fewer than LIMIT characters, when over limit.
    size = LIMIT + extra_bytes
    payload = "é" * (size // 2) + "x" * (size % 2)
    original, blanked = oversized_html(payload)
    assert len(payload.encode()) == size and len(payload) < LIMIT
    assert projected_bytes(original) == (blanked if extra_bytes > 0 else original).encode()


@pytest.mark.parametrize("html", [
    '<p>Uni<script data-x="1">small</script>ted States</p>',
    '<style>.hidden{display:none}</style><script type="application/ld+json">{"name":"Baringa"}</script>',
    '<!-- <script>not an element</script> --><p>Body</p>',
    '<script/>not a complete script element',
])
def test_small_html_preserves_every_original_byte(html):
    assert worker_html(document(html)).encode() == html.encode()


def test_inline_boundaries_styles_hidden_links_and_other_bytes_are_preserved():
    payload = "x" * (LIMIT + 1)
    original = ('<style>.hide{display:none}</style>Uni<script id="same">' + payload
        + '</script>ted <a class="hide" href="' + BOARD_LINK + '">Hidden</a>'
        + '<a href="https://job-boards.eu.greenhouse.io/another/jobs/1234567890">Apply</a>')
    expected = original.replace(payload, "")
    projected = worker_html(document(original))
    assert projected == expected
    assert visible_html_text(projected) == visible_html_text(original)
    assert "Uni ted" in visible_html_text(projected)
    assert list(visible_html_link_labels(projected)) == list(visible_html_link_labels(original))
    assert anchors(projected) == []  # Hidden matching tenant and visible wrong tenant grant nothing.
    assert anchors(projected, source="https://other.example/careers") == []


def test_each_script_uses_the_requested_operation_cap(monkeypatch):
    operation = operations.OPERATIONS["scrapingdog.scrape"]
    monkeypatch.setattr(operations, "OPERATIONS", {
        **operations.OPERATIONS, "scrapingdog.scrape": replace(operation, max_response_bytes=256),
    })
    prefix = '<head>é\r\n<script type="application/ld+json">small</script>\n'
    original = prefix + '<script id="one">' + "x" * 257 + '</script>\r\n'
    original += '<SCRIPT id="two">' + "é" * 129 + '</SCRIPT>\n</head><body>Jobs</body>'
    expected = prefix + '<script id="one"></script>\r\n<SCRIPT id="two"></SCRIPT>\n</head><body>Jobs</body>'
    assert worker_html(document(original)).encode() == expected.encode()


@pytest.mark.parametrize("template", [
    '<script>{payload}',
    '<script type="unterminated>{payload}</script><body>Jobs</body>',
    '<script>{payload}</script broken><body>Jobs</body>',
    '<!-- <script>{payload}</script> --><body>Jobs</body>',
    '<style>{payload}</style><body>Jobs</body>',
    '<body>{payload}</body>',
    '<script/>{payload}<body>Jobs</body>',
])
def test_unclosed_malformed_or_non_script_content_is_not_removed(template):
    original = template.format(payload="x" * (LIMIT + 1))
    assert projected_bytes(original) == original.encode()
    assert worker_html(document(original)).encode() == operations._truncate_utf8(original.encode(), LIMIT)


@pytest.mark.parametrize("error", [AssertionError, ValueError])
def test_parser_failure_does_not_apply_partial_spans(monkeypatch, error):
    def reject_close(_parser):
        raise error("malformed marked section")

    monkeypatch.setattr(operations.HTMLParser, "close", reject_close)
    original, _ = oversized_html()
    assert projected_bytes(original) == original.encode()
    assert worker_html(document(original)).encode() == operations._truncate_utf8(original.encode(), LIMIT)


def test_several_individually_small_scripts_are_not_removed():
    script = '<script>' + "x" * (LIMIT // 2) + '</script>'
    original = '<head>' + script * 3 + '</head><body>Jobs</body>'
    assert projected_bytes(original) == original.encode()
    assert worker_html(document(original)).encode() == operations._truncate_utf8(original.encode(), LIMIT)


@pytest.mark.parametrize("metadata,code", [
    ({"sourceURL": "https://other.example/"}, "firecrawl_source_url_mismatch"),
    ({"url": "http://baringa.com/"}, "invalid_firecrawl_final_url"),
    ({"statusCode": True}, "missing_firecrawl_target_status"),
    ({"contentType": "application/pdf"}, "missing_firecrawl_raw_html"),
])
def test_oversized_projection_never_bypasses_existing_source_and_pdf_guards(metadata, code):
    original, _ = oversized_html()
    with pytest.raises(compat.CompatibilityResponseError, match=code):
        adapted(document(original, **metadata))


@pytest.mark.parametrize("prefix", ["%PDF-1.7", "JVBERi0"])
def test_document_payload_cannot_become_html_after_projection(prefix):
    original, _ = oversized_html()
    with pytest.raises(compat.CompatibilityResponseError, match="missing_firecrawl_raw_html"):
        adapted(document(prefix + original))


@pytest.mark.parametrize("status", [200, 301, 404, 410])
def test_target_status_and_final_url_are_not_changed(status):
    original, expected = oversized_html()
    final = "https://www.baringa.com/actual-response/"
    result = adapted(document(original, statusCode=status, url=final))
    assert result[0] == status and result[3] == final
    assert result[2] == original.encode()
    sanitized = operations.sanitize_response("scrapingdog.scrape", *result[:3])
    assert sanitized[0] == status
    assert sanitized[2] == (expected.encode() if 200 <= status < 300
                            else operations._truncate_utf8(original.encode(), LIMIT))


@pytest.mark.parametrize("status", [403, 429, 503])
def test_infrastructure_target_retains_existing_unavailable_path(status):
    original, _ = oversized_html()
    with pytest.raises(compat.CompatibilityResponseError, match="firecrawl_source_content_unavailable") as error:
        adapted(document(original, statusCode=status))
    assert error.value.trusted_response_url == SOURCE


def test_exact_retained_public_provider_reply():
    source = os.environ.get("ARENA_OVERSIZED_SCRIPT_RESPONSE")
    if not source:
        pytest.skip("Exact retained public provider response is opt-in")
    retained = Path(source).read_bytes()
    assert hashlib.sha256(retained).hexdigest() == "d7693836bd731425f39f9dfc5792d25ad9365afcd16ca015020f20503ce64e64"
    raw = json.loads(retained)
    html = raw["result"]["data"]["rawHtml"]
    before = list(visible_html_link_labels(html))
    projected = worker_html(raw)
    # Exact independently recorded character offsets in this bound response.
    assert html[4315725:4315734] == "</script>"
    assert projected == html[:87065] + html[4315725:]
    assert list(visible_html_link_labels(projected)) == before
    assert len(before) == 162
    assert sum("job-boards.eu.greenhouse.io/baringa/jobs/" in href for href, _ in before) == 9
    assert anchors(projected) == anchors(html)  # The existing ownership policy is unchanged.
    assert anchors(projected, posting="https://job-boards.eu.greenhouse.io/another/jobs/1234567890") == []
    assert len(projected.encode()) < LIMIT


@pytest.mark.parametrize("content_type", ["text/html; charset=UTF-8", "application/xhtml+xml"])
def test_direct_native_html_uses_the_same_projection(content_type):
    original, expected = oversized_html("x" * (LIMIT + 1))
    status, headers, body = operations.sanitize_response(
        "scrapingdog.scrape", 200, {"content-type": content_type}, original.encode())
    assert status == 200 and body == expected.encode()
    assert headers["content-length"] == str(len(body))


@pytest.mark.parametrize("status", [201, 204, 299])
def test_success_status_is_preserved_by_shared_html_projection(status):
    original, expected = oversized_html("x" * (LIMIT + 1))
    result = operations.sanitize_response("scrapingdog.scrape", status,
        {"content-type": "text/html"}, original.encode())
    assert result[0] == status and result[2] == expected.encode()


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/json", "application/pdf", "text/html invalid"])
def test_non_html_content_keeps_the_existing_text_cap(content_type):
    original, _ = oversized_html("x" * (LIMIT + 1))
    headers = {} if content_type is None else {"content-type": content_type}
    result = operations.sanitize_response("scrapingdog.scrape", 200, headers, original.encode())
    assert result[2] == operations._truncate_utf8(original.encode(), LIMIT)


@pytest.mark.parametrize("status", [100, 301, 400, 404, 410, 422])
def test_non_success_html_keeps_the_existing_text_cap(status):
    original, _ = oversized_html("x" * (LIMIT + 1))
    result = operations.sanitize_response("scrapingdog.scrape", status,
        {"content-type": "text/html"}, original.encode())
    assert result[0] == status
    assert result[2] == operations._truncate_utf8(original.encode(), LIMIT)


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_infrastructure_html_still_returns_only_the_generic_unavailable_body(status):
    original, _ = oversized_html("x" * (LIMIT + 1))
    result = operations.sanitize_response("scrapingdog.scrape", status,
        {"content-type": "text/html"}, original.encode())
    assert result[0] == operations.GENERIC_UNAVAILABLE_STATUS
    assert result[2] == operations.GENERIC_UNAVAILABLE_BODY


@pytest.mark.parametrize("invalid", [b"\xff", b"\xc3"])
def test_invalid_utf8_uses_old_cap_without_replacement_or_partial_projection(invalid):
    original, _ = oversized_html("x" * (LIMIT + 1))
    body = invalid + original.encode()
    assert operations._project_oversized_html_scripts(body, LIMIT) == body
    result = operations.sanitize_response("scrapingdog.scrape", 200,
        {"content-type": "text/html"}, body)
    assert result[2] == operations._truncate_utf8(body, LIMIT)


def test_json_operation_does_not_project_script_strings_or_change_its_cap():
    original, _ = oversized_html("x" * (LIMIT + 1))
    body = json.dumps({"results": [{"text": original}]}).encode()
    with pytest.raises(operations.OperationResponseError, match="response_too_large"):
        operations.sanitize_response("exa.contents", 200,
            {"content-type": "text/html"}, body)


def test_generic_http_source_adapter_reaches_the_shared_projection():
    from tests.lab_arena import generic_html_source_adapter_test as generic

    original, expected = oversized_html("x" * (LIMIT + 1))
    reply = generic.envelope()
    reply["toolResponse"]["rawV2"]["data"] = original
    status, headers, body, final = generic.adapt(reply)
    assert body == original.encode() and final == generic.FINAL
    result = operations.sanitize_response("scrapingdog.scrape", status, headers, body)
    assert result[2] == expected.encode()
    # Response sanitation neither supplies nor changes trusted URL authority.
    assert set(result[1]) == {"content-type", "content-length"}


def test_native_transport_ceiling_is_not_increased_for_the_retained_size():
    import httpx
    from lab_arena import broker
    from tests.lab_arena.test_lab_arena_broker import _async_client_factory

    original, _ = oversized_html()
    assert len(original.encode()) > 4 * 1024 * 1024
    transport = broker.HttpxProviderTransport(client_factory=_async_client_factory(
        lambda _request: httpx.Response(200, headers={"content-type": "text/html"}, content=original.encode())))
    try:
        result = transport.send(method="GET", url="https://api.scrapingdog.com/scrape",
            headers={}, body=b"", timeout_seconds=1)
    finally:
        transport.close()
    assert result.status == 502 and result.internal_provenance == "response_too_large"
    assert result.body == operations.GENERIC_UNAVAILABLE_BODY
