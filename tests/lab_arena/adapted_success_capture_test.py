"""Successful compatibility adaptation preserves private provider evidence."""
import base64
import json

from tests.lab_arena.score_compat_capability_test import (
    firecrawl_catalog, routed_broker, score_context,
)
from tests.lab_arena.test_lab_arena_broker import FakeTransport


def test_adapted_success_retains_source_metadata_without_changing_replay():
    source = "https://example.com/about"
    document = {
        "status": "completed", "billing": {"credits_charged": 0.02},
        "result": {"data": {"rawHtml": "<html>Company facts</html>",
            "metadata": {"url": source, "sourceURL": source,
                "statusCode": 200, "modifiedTime": "2026-11-06",
                "author": "Example Author", "api_key": "private-test-secret"}}},
    }
    transport = FakeTransport([(200, document)])
    broker, store, _ = routed_broker(transport=transport)
    kwargs = dict(operation_id="scrapingdog.scrape", parameters={"url": source},
                  action_sequence=1, timeout_ms=30_000)
    first = broker.execute(score_context(firecrawl_catalog()), **kwargs)
    saved = store.calls[first.call["call_identity"]]["terminal"]
    receipt = saved["upstream_response_receipt"]
    retained = base64.b64decode(receipt["body_b64"])
    metadata = json.loads(retained)["result"]["data"]["metadata"]
    assert metadata["modifiedTime"] == "2026-11-06"
    assert metadata["author"] == "Example Author"
    assert b"private-test-secret" not in retained
    assert receipt["privacy"]["redacted"] is True
    assert receipt["call_succeeded"] is True
    assert first.status == 200 and first.body == b"<html>Company facts</html>"
    assert "upstream_response_receipt" not in json.dumps(first.to_document())
    repeated = broker.execute(score_context(firecrawl_catalog()), **kwargs)
    assert (repeated.status, repeated.headers, repeated.body) == (first.status, first.headers, first.body)
    assert len(transport.sent) == 1


def test_identity_adapter_does_not_duplicate_upstream_body(monkeypatch):
    from lab_arena import scoring_provider_compat
    monkeypatch.setattr(scoring_provider_compat, "adapt_response_with_trusted_url",
        lambda route, *, status, headers, body: (status, headers, body, ""))
    broker, store, _ = routed_broker(transport=FakeTransport([(200, {
        "status": "completed", "billing": {"credits_charged": 0.02},
        "result": {"data": {"rawHtml": "<html>facts</html>"}},
    })]))
    result = broker.execute(score_context(firecrawl_catalog()),
        operation_id="scrapingdog.scrape", parameters={"url": "https://example.com/"},
        action_sequence=1, timeout_ms=30_000)
    assert result.status == 200
    assert "upstream_response_receipt" not in store.calls[result.call["call_identity"]]["terminal"]
