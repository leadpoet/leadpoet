"""Successful compatibility adaptation preserves private provider evidence."""
import base64
import json

import pytest

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


def test_duplicate_tool_metadata_cannot_bypass_private_capture_redaction():
    document = b'{"job_id":"fake-job-1","status":"completed","billing":{"credits_charged":0.02},"result":{"data":{"rawHtml":"<html>safe facts</html>","metadata":{"url":"https://example.com/","sourceURL":"https://example.com/","statusCode":200,"api_key":"synthetic-private-test-value","api_key":null}}}}'
    transport = FakeTransport([(200, document)])
    broker, store, _ = routed_broker(transport=transport)
    kwargs = dict(operation_id="scrapingdog.scrape", parameters={"url": "https://example.com/"},
                  action_sequence=1, timeout_ms=30_000)
    first = broker.execute(score_context(firecrawl_catalog()), **kwargs)
    saved = store.calls[first.call["call_identity"]]["terminal"]
    assert first.status == 200 and first.body == b"<html>safe facts</html>"
    assert saved["upstream_response_receipt"] == {
        "status": "unavailable", "reason": "response_capture_failed",
    }
    assert "synthetic-private-test-value" not in json.dumps(saved)
    repeated = broker.execute(score_context(firecrawl_catalog()), **kwargs)
    assert (repeated.status, repeated.headers, repeated.body) == (first.status, first.headers, first.body)
    assert len(transport.sent) == 1


@pytest.mark.parametrize("body", [
    b'{"api_key":"synthetic-secret","api_key":null}',
    b'[{"metadata":{"api_key":"synthetic-secret","api_key":null}}]',
    b'{"metadata":{"api_key":"synthetic-secret","api_\\u006bey":null}}',
])
@pytest.mark.parametrize("status", [200, 503])
def test_duplicate_tool_json_is_not_reinterpreted_as_plain_text(body, status):
    from lab_arena import broker as br
    receipt = br._capture_upstream_response(br.ProviderResponse(status, {}, body),
        operation_id="scrapingdog.scrape", request_hash="hash", call_identity="call")
    assert receipt == {"status": "unavailable", "reason": "response_capture_failed"}


@pytest.mark.parametrize("body", [
    b'[{"facts":"safe"}]', b'"safe text"', b'123', b'null',
    b'<html>safe facts</html>', b'{unfinished public text',
])
def test_nonobject_json_and_plain_tool_text_remain_capturable(body):
    from lab_arena import broker as br
    receipt = br._capture_upstream_response(br.ProviderResponse(200, {}, body),
        operation_id="scrapingdog.scrape", request_hash="hash", call_identity="call")
    assert base64.b64decode(receipt["body_b64"]) == body
    assert receipt["privacy"]["redacted"] is False


def test_tool_json_array_still_redacts_nested_credentials():
    from lab_arena import broker as br
    body = b'[{"metadata":{"api_key":"synthetic-secret","source":"public"}}]'
    receipt = br._capture_upstream_response(br.ProviderResponse(200, {}, body),
        operation_id="scrapingdog.scrape", request_hash="hash", call_identity="call")
    assert json.loads(base64.b64decode(receipt["body_b64"])) == [
        {"metadata": {"api_key": "[REDACTED]", "source": "public"}}]
    assert receipt["privacy"]["redacted"] is True


@pytest.mark.parametrize("capture_case", ["credential", "size"])
def test_optional_success_capture_preserves_eligibility_and_sft_targets(capture_case):
    from copy import deepcopy
    from lab_arena import training_thinking
    from lab_arena.training_export import assemble_example
    from tests.lab_arena.training_export_test import fixture, add_call
    from tests.lab_arena.training_eligibility_test import captured_example, capture
    from tests.lab_arena.training_steps_test import seal
    from tests.lab_arena.training_thinking_test import full_example

    source = "https://example.com/about"
    metadata = {"url": source, "sourceURL": source, "statusCode": 200}
    metadata.update({"api_key": "private-test-secret"} if capture_case == "credential"
                    else {"provider_padding": "x" * (3 * 1024 * 1024)})
    broker, store, _ = routed_broker(transport=FakeTransport([(200, {
        "status": "completed", "billing": {"credits_charged": 0.02},
        "result": {"data": {"rawHtml": "<html>facts</html>", "metadata": metadata}},
    })]))
    result = broker.execute(score_context(firecrawl_catalog()),
        operation_id="scrapingdog.scrape", parameters={"url": source},
        action_sequence=1, timeout_ms=30_000)
    assert result.status == 200 and result.body == b"<html>facts</html>"
    retained = store.calls[result.call["call_identity"]]["terminal"]["upstream_response_receipt"]
    values = fixture()
    add_call(values, provider="deepline", operation="scrapingdog.scrape")
    before = assemble_example(**values)["provider_calls"][0]
    reserve = values["ledger_rows"][0]
    receipt = dict(retained)
    if receipt.get("status") != "unavailable":
        receipt.update(operation_id="scrapingdog.scrape", call_identity=reserve["call_identity"],
                       request_hash=reserve["entry_doc"]["request_hash"])
    values["ledger_rows"][-1]["terminal_response"]["upstream_response_receipt"] = receipt
    after = assemble_example(**values)["provider_calls"][0]
    gaps = set(after["evidence_gaps"]) - set(before["evidence_gaps"])
    assert gaps == ({"upstream_response_privacy_nonexact"} if capture_case == "credential"
                    else {"upstream_response_capture_unavailable", "upstream_response_size_limit"})

    def example_with_tool(factory, tool, diagnostic_gaps=()):
        example = factory()
        example["provider_calls"].append(dict(deepcopy(tool), run_id=example["run_id"],
                                              call_identity="unused-tool-evidence"))
        proof = example["completeness"]
        if diagnostic_gaps:
            prior = proof["reasons_by_dimension"].get("research_process", [])
            proof["reasons_by_dimension"]["research_process"] = sorted(set(prior) | set(diagnostic_gaps))
            proof["dimensions"]["research_process"] = False
            proof["incomplete_reasons"] = sorted(set(proof["incomplete_reasons"]) | set(diagnostic_gaps))
            proof["complete"] = False
        return seal(example)

    old = capture(example_with_tool(captured_example, before))
    new = capture(example_with_tool(captured_example, after, gaps))
    assert old["eligible"] is True and new == old
    old_rows = training_thinking.build([example_with_tool(full_example, before)])[0]
    new_rows = training_thinking.build([example_with_tool(full_example, after, gaps)])[0]
    old_targets = [row["target"] for split in ("train", "eval") for row in old_rows[split]]
    new_targets = [row["target"] for split in ("train", "eval") for row in new_rows[split]]
    assert old_targets and new_targets == old_targets
