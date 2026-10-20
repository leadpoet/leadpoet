"""Credential-free upstream throttles must keep provider recovery authority."""

import json
import os

import pytest

from lab_arena import broker as br, contracts, runner, runtime, shim
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, lease, make_config,
)


PARAMETERS = {
    "tool": "generic_http_request", "payload": {"url": "https://example.com/"},
}


def upstream_rate_limit(provider="generic_http", operation="generic_http_request"):
    # Non-secret structured fields from the authenticated October 20 response.
    return {
        "code": "UPSTREAM_RATE_LIMIT",
        "credential_owner": "none", "credential_source": "none",
        "failure_origin": "provider", "error_category": "upstream",
        "provider": provider, "operation": operation, "upstream_status": 429,
        "rate_limit_kind": "upstream_provider",
        "rate_limit_bucket_scope": "provider_shared",
        "rate_limit_credential_source": "none", "rate_limit_effective_limit": 25,
        "tool_error": {
            "code": "UPSTREAM_RATE_LIMIT", "provider": provider,
            "operation": operation, "statusCode": 429,
            "category": "rate_limit", "origin": "provider", "retryable": True,
        },
    }


def response(document, status=429, **kwargs):
    return br.ProviderResponse(
        status, {"content-type": "application/json"},
        json.dumps(document).encode(), **kwargs,
    )


@pytest.mark.parametrize("provider,operation", [
    ("generic_http", "generic_http_request"), ("another_provider", "another_read"),
])
def test_structured_credential_free_rate_limit_is_provider_failure(provider, operation):
    saved = response(upstream_rate_limit(provider, operation))
    for champion in (False, True):
        assert not br._miner_credential_failure(
            "deepline", saved, champion_credential_retry=champion,
        )
    assert br._miner_credential_failure(
        "openrouter", saved, champion_credential_retry=True,
    )


@pytest.mark.parametrize("field,value", [
    ("code", "INSUFFICIENT_CREDITS"), ("credential_owner", "customer"),
    ("credential_owner", "deepline_managed"), ("credential_owner", None),
    ("credential_source", "customer"), ("credential_source", None),
    ("failure_origin", "account"), ("error_category", "provider_account"),
    ("upstream_status", "429"), ("upstream_status", 401),
    ("provider", ""), ("operation", ""), ("tool_error", {}),
])
def test_ambiguous_or_customer_rate_limit_remains_account_failure(field, value):
    document = upstream_rate_limit()
    document[field] = value
    assert br._miner_credential_failure(
        "deepline", response(document), champion_credential_retry=True,
    )


@pytest.mark.parametrize("field,value", [
    ("code", "OTHER"), ("provider", "other"), ("operation", "other"),
    ("statusCode", 401), ("statusCode", "429"),
    ("category", "account"), ("origin", "account"),
])
def test_inconsistent_nested_rate_limit_cannot_exempt_account_failure(field, value):
    document = upstream_rate_limit()
    document["tool_error"][field] = value
    assert br._miner_credential_failure(
        "deepline", response(document), champion_credential_retry=True,
    )


@pytest.mark.parametrize("field", [
    "rate_limit_kind", "rate_limit_bucket_scope", "rate_limit_credential_source",
])
@pytest.mark.parametrize("change", ["missing", "null", "conflicting"])
def test_rate_limit_attribution_must_be_complete_and_consistent(field, change):
    document = upstream_rate_limit()
    if change == "missing":
        document.pop(field)
    else:
        document[field] = None if change == "null" else "customer_account"
    assert br._miner_credential_failure(
        "deepline", response(document), champion_credential_retry=True,
    )


@pytest.mark.parametrize("status", [401, 402, 403])
def test_real_authentication_and_credit_status_keep_account_failure(status):
    assert br._miner_credential_failure(
        "deepline", response(upstream_rate_limit(), status),
        champion_credential_retry=True,
    )


@pytest.mark.parametrize("document", [None, [], {"error": "rate limited"}])
def test_unstructured_rate_limit_remains_conservative(document):
    assert br._miner_credential_failure(
        "deepline", response(document), champion_credential_retry=True,
    )


@pytest.mark.parametrize("body", [b"not-json", b"\xff"])
def test_malformed_rate_limit_remains_conservative(body):
    saved = br.ProviderResponse(429, {"content-type": "application/json"}, body)
    assert br._miner_credential_failure(
        "deepline", saved, champion_credential_retry=True,
    )


def test_internal_response_cannot_exempt_account_failure():
    assert br._miner_credential_failure(
        "deepline", response(upstream_rate_limit(), internal_provenance="credential_echo"),
        champion_credential_retry=True,
    )


def rate_limit_broker(*, bill=None):
    marked = []
    document = upstream_rate_limit()
    if bill is not None:
        document["billing"] = {"credits_charged": bill}
        document["requestId"] = "public-execute:00000000-0000-4000-8000-000000000001"
    broker, store, transport = make_broker(
        transport=FakeTransport([(429, document)]),
        provider_funding_source_for=lambda *_: "miner_key",
        retry_miner_credential_for=lambda _: True,
        mark_provider_fallback=lambda *args: marked.append(args),
    )
    return broker, store, transport, marked


def execute_rate_limit(*, bill=None):
    broker, store, transport, marked = rate_limit_broker(bill=bill)
    result = broker.execute(
        CONTEXT, operation_id="deepline.execute", parameters=PARAMETERS,
        action_sequence=95, timeout_ms=5000,
    )
    assert result.status == 502
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["provider_status"] == 429
    assert result.call["provider_attempt"] == 1
    assert sum(x["method"] == "POST" for x in transport.sent) == 1
    assert marked == []
    assert "account_failure_evidence" not in json.dumps(store.calls)
    assert "credit_failure_proof" not in json.dumps(store.calls)
    return result, store


def test_champion_throttle_keeps_unknown_billing_without_account_retry_or_fallback():
    result, store = execute_rate_limit()
    assert result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    saved = store.calls[result.call["call_identity"]]
    assert saved["uncertain_doc"]["call_succeeded"] is False
    assert saved["uncertain_doc"]["provider_status"] == 429


@pytest.mark.parametrize("bill,expected_microusd", [(0, 0), (0.02, 2000)])
def test_champion_throttle_retains_exact_billing(bill, expected_microusd):
    result, store = execute_rate_limit(bill=bill)
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == expected_microusd
    saved = store.calls[result.call["call_identity"]]
    assert saved["actual"] == expected_microusd
    assert saved["terminal"]["call_succeeded"] is False


def test_upstream_throttle_does_not_poison_the_submitted_credential():
    broker, store, transport, marked = rate_limit_broker()
    first = broker.execute(
        CONTEXT, operation_id="deepline.execute", parameters=PARAMETERS,
        action_sequence=95, timeout_ms=5000,
    )
    transport.responses.append((200, {"result": {"data": {"text": "page"}}}))
    second = broker.execute(
        CONTEXT, operation_id="deepline.execute", parameters=PARAMETERS,
        action_sequence=96, timeout_ms=5000,
    )
    assert first.status == 502 and second.status == 200
    assert first.call["call_identity"] != second.call["call_identity"]
    assert first.call["outcome"] == "uncertain"
    assert second.call["outcome"] == "settled"
    assert sum(x["method"] == "POST" for x in transport.sent) == 2
    assert marked == []
    assert "account_failure_evidence" not in json.dumps(store.calls)


def test_real_broker_worker_runner_transition_keeps_provider_recovery(tmp_path):
    broker, store, transport, marked = rate_limit_broker()

    class BrokerApi(FakeApi):
        def provider(self, _run_id, _lease_token, frame):
            self.provider_frames.append(dict(frame))
            return broker.execute(CONTEXT, **frame).to_document()

    class ThrottledRuntime(BridgingRuntime):
        def run_icp(self, spec, **_):
            self.specs.append(spec)
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                status, _headers, body = shim.dispatch(
                    "deepline.execute", PARAMETERS, 5000,
                )
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            assert status == 502
            assert json.loads(body) == {"error": {"code": "provider_unavailable"}}
            return runtime.fake_result(exit_code=1, output_bytes=None)

    api = BrokerApi([lease()])
    sandbox = ThrottledRuntime()
    (tmp_path / "work").mkdir()
    worker = runner.Runner(make_config(tmp_path, api, sandbox))
    try:
        assert worker.run_once() == 1
    finally:
        worker.close()
    result = contracts.validate_run_result(api.completions[0]["body"]["result"])
    assert result["terminal_status"] == "provider_error"
    assert result["resource_summary"]["provider_call_count"] == 1
    assert api.completions[0]["body"]["output"] is None
    assert worker.abandoned == 0
    assert sum(x["method"] == "POST" for x in transport.sent) == 1
    assert marked == []
    assert "account_failure_evidence" not in json.dumps(store.calls)
    assert not sandbox.specs[0].input_dir.exists()
    assert not sandbox.specs[0].agent_entrypoint_path.exists()
