"""Typed pre-generation budget throttles must not become miner credit failures."""

import json

import pytest

from lab_arena import broker as br
from lab_arena import runner
from tests.lab_arena.test_lab_arena_broker import CHAT, CONTEXT, FakeTransport, HOST_KEYS, make_broker
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    ReviewTransport, _archive, _reviewer, _source,
)


def budget_error(source="openrouter_in_flight_budget"):
    return {"error": {"code": 402, "message": "budget unavailable", "metadata": {"limit_source": source}}}


@pytest.mark.parametrize("operation,parameters", [
    ("openrouter.chat", CHAT),
    ("openrouter.responses", {"model": "openai/gpt-4o-mini", "input": "research"}),
])
@pytest.mark.parametrize("champion", [False, True])
def test_budget_throttle_uses_existing_retry_path_without_champion_fallback(operation, parameters, champion):
    fallback = []
    gate = br.OpenRouterSharedGate(max_concurrency=1)
    instance, _store, transport = make_broker(
        transport=FakeTransport([(402, budget_error(), {"retry-after": "30"})]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "miner_key",
        retry_miner_credential_for=lambda _context: champion,
        mark_provider_fallback=lambda *_args: fallback.append(True),
        openrouter_shared_gate=gate,
    )
    result = instance.execute(CONTEXT, operation_id=operation, parameters=parameters,
                              action_sequence=1, timeout_ms=120_000)
    assert result.status == 502
    assert result.call["error_code"] == "provider_unavailable"
    assert result.call["provider_status"] == 429
    assert result.call["openrouter_original_http_status"] == 402
    assert result.call["actual_microusd"] == 0
    assert result.call["outcome"] == "settled"
    assert "credit_failure_proof" not in result.call
    assert not fallback
    assert len(transport.sent) == 1
    if operation == "openrouter.responses":
        assert result.call["retry_after_seconds"] == 30
        assert 30 <= runner.WorkerSocketServer._responses_rate_limit_delay(result.call, 0) <= 32


@pytest.mark.parametrize("source", ["openrouter_key_limit", "openrouter_credits", "unknown", None])
def test_other_credit_errors_remain_terminal(source):
    document = budget_error(source)
    response = br.ProviderResponse(402, {}, json.dumps(document).encode())
    assert br._openrouter_effective_response(response) is response
    assert br._miner_credential_failure("openrouter", response, champion_credential_retry=True)


@pytest.mark.parametrize("extra", [
    {"id": "gen-existing"}, {"usage": {"cost": "0.01"}},
    {"usage": None}, {"choices": []}, {"output": []},
])
def test_generation_or_usage_markers_prevent_budget_reclassification(extra):
    response = br.ProviderResponse(402, {}, json.dumps({**budget_error(), **extra}).encode())
    assert br._openrouter_effective_response(response) is response


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"error":{"code":402}}'])
def test_malformed_response_is_not_a_budget_throttle(body):
    response = br.ProviderResponse(402, {}, body)
    assert br._openrouter_effective_response(response) is response


def test_embedded_402_is_not_pre_generation_budget_proof():
    response = br.ProviderResponse(200, {}, json.dumps(budget_error()).encode())
    effective = br._openrouter_effective_response(response)
    assert effective.status == 402
    assert br._miner_credential_failure("openrouter", effective, champion_credential_retry=False)


def test_synthetic_response_does_not_gain_budget_authority():
    response = br.ProviderResponse(402, {}, json.dumps(budget_error()).encode(), "credential_echo")
    assert br._openrouter_effective_response(response) is response


@pytest.mark.parametrize("value", ["gen-existing", "invalid generation id"])
def test_header_generation_identity_prevents_budget_reclassification(value):
    response = br.ProviderResponse(402, {"X-Generation-Id": value}, json.dumps(budget_error()).encode())
    assert br._openrouter_effective_response(response) is response


def test_conflicting_wire_status_does_not_gain_champion_budget_exemption():
    response = br.ProviderResponse(429, {}, json.dumps(budget_error()).encode())
    assert br._miner_credential_failure(
        "openrouter", response, champion_credential_retry=True, original_http_status=429,
    )


@pytest.mark.parametrize("source,code,retryable", [
    ("openrouter_in_flight_budget", "code_review_provider_rate_limited", True),
    ("openrouter_credits", "code_review_provider_credit", False),
    ("unknown", "code_review_provider_credit", False),
])
def test_review_accounts_complete_direct_refusal_and_keeps_wire_status(source, code, retryable):
    transport = ReviewTransport(lambda _parameters: {"status": 402, "body": budget_error(source)})
    reviewer, row, store, _objects = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["error_code"] == code
    assert document["retryable"] is retryable
    assert document["provider_http_status"] == 402
    assert store.finish_calls[0][5] == 0


@pytest.mark.parametrize("status,extra,expected_cost", [
    (200, {}, None),
    (402, {"id": "gen-existing"}, None),
    (402, {"usage": None}, None),
    (402, {"usage": {"cost": "0.01"}}, 10000),
    (402, {"choices": []}, None),
])
def test_review_preserves_generation_uncertainty_and_confirmed_charges(status, extra, expected_cost):
    transport = ReviewTransport(lambda _parameters: {"status": status, "body": {**budget_error(), **extra}})
    reviewer, row, store, _objects = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    assert store.finish_calls[0][5] == expected_cost


def test_review_transport_failure_remains_uncertain():
    class Transport:
        def send(self, **_kwargs):
            raise br.ProviderTransportError("ReadTimeout")
    reviewer, row, store, _objects = _reviewer(_archive(_source()), Transport())
    assert reviewer.review(row)["status"] == "error"
    assert store.finish_calls[0][4]["error_code"] == "code_review_transport_error"
    assert store.finish_calls[0][5] is None


def test_malformed_review_refusal_remains_uncertain():
    transport = ReviewTransport(lambda _parameters: {"status": 402, "body": {"error": {"code": 402}}})
    reviewer, row, store, _objects = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    assert store.finish_calls[0][5] is None


def test_review_header_generation_keeps_uncertain_charge_and_credit_classification():
    class Transport:
        def send(self, **_kwargs):
            return br.ProviderResponse(
                402, {"x-generation-id": "gen-existing"}, json.dumps(budget_error()).encode(),
            )
    reviewer, row, store, _objects = _reviewer(_archive(_source()), Transport())
    assert reviewer.review(row)["status"] == "error"
    assert store.finish_calls[0][4]["error_code"] == "code_review_provider_credit"
    assert store.finish_calls[0][5] is None
