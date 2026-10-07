"""A managed provider outage must not invalidate the miner's Deepline account."""
import json

import pytest

from lab_arena import broker as br
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeTransport, make_broker,
)


def managed_error(provider="contextdev", operation="contextdev_get_web_scrape_markdown"):
    # Typed, non-secret fields from the saved October 7 provider response.
    return {
        "code": "PROVIDER_ACCOUNT_CAPACITY",
        "credential_owner": "deepline_managed", "credential_source": "env",
        "error_category": "provider_account", "failure_origin": "provider_account",
        "operation": operation, "provider": provider, "upstream_status": 401,
        "tool_error": {"code": "PROVIDER_ACCOUNT_CAPACITY", "operation": operation,
                       "provider": provider, "statusCode": 402},
    }


def response(document, status=402):
    return br.ProviderResponse(status, {"content-type": "application/json"},
                               json.dumps(document).encode())


@pytest.mark.parametrize("provider,operation", [
    ("contextdev", "contextdev_get_web_scrape_markdown"),
    ("another_research_provider", "another_read_operation"),
])
def test_managed_account_scope_is_provider_independent(provider, operation):
    saved = response(managed_error(provider, operation))
    for champion in (False, True):
        assert not br._miner_credential_failure(
            "deepline", saved, champion_credential_retry=champion)
    assert br._miner_credential_failure(
        "openrouter", saved, champion_credential_retry=False)
    assert br._confirmed_credit_failure_proof(
        provider="deepline", funding_source="miner_key", response=saved,
        raw_document=managed_error(provider, operation), raw_actual=0, actual=0,
        call_succeeded=False, openrouter_generation_present=False) is None


@pytest.mark.parametrize("field,value", [
    ("credential_owner", "customer"), ("credential_owner", None),
    ("failure_origin", "account"), ("error_category", "unknown"),
    ("upstream_status", "401"), ("upstream_status", 200),
    ("provider", ""), ("operation", ""), ("code", ""), ("tool_error", {}),
])
def test_ambiguous_or_customer_account_errors_keep_normal_refusal(field, value):
    document = managed_error()
    document[field] = value
    assert not br._deepline_managed_provider_failure(response(document))
    assert br._miner_credential_failure(
        "deepline", response(document), champion_credential_retry=True)


@pytest.mark.parametrize("field,value", [("provider", "other"),
    ("operation", "different"), ("code", "INSUFFICIENT_CREDITS"),
    ("statusCode", 401), ("statusCode", "402")])
def test_conflicting_nested_error_is_not_managed_account_proof(field, value):
    document = managed_error()
    document["tool_error"][field] = value
    assert not br._deepline_managed_provider_failure(response(document))


@pytest.mark.parametrize("champion", [False, True])
def test_broker_preserves_unknown_bill_without_marking_miner_or_retrying_account(champion):
    marked = []
    broker, store, transport = make_broker(
        transport=FakeTransport([(402, managed_error())]),
        provider_funding_source_for=lambda *_: "miner_key",
        retry_miner_credential_for=lambda _: champion,
        mark_provider_fallback=lambda *args: marked.append(args),
    )
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute",
        parameters={"tool": "contextdev_get_web_scrape_markdown",
                    "payload": {"url": "https://example.com/"}},
        action_sequence=65, timeout_ms=5000)
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    saved = store.calls[result.call["call_identity"]]
    assert saved["uncertain_doc"]["call_succeeded"] is False
    assert "account_failure_evidence" not in saved["uncertain_doc"]
    assert "credit_failure_proof" not in saved["uncertain_doc"]
    assert sum(x["method"] == "POST" for x in transport.sent) == 1
    assert marked == []


def test_real_account_credit_refusal_keeps_credit_proof():
    saved = response({"error": "Insufficient credits"})
    assert br._miner_credential_failure("deepline", saved, champion_credential_retry=True)
    proof = br._confirmed_credit_failure_proof(
        provider="deepline", funding_source="miner_key", response=saved,
        raw_document={"error": "Insufficient credits"}, raw_actual=0, actual=0,
        call_succeeded=False, openrouter_generation_present=False)
    assert proof["reason"] == "out_of_credit"
