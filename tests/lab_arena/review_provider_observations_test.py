"""Private review observations do not become retry or billing authority."""

import json

import pytest

from lab_arena import code_review_policy
from tests.lab_arena.openrouter_in_flight_budget_test import budget_error
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, ReviewTransport, _archive, _reviewer, _source,
)


@pytest.mark.parametrize("source,code,retryable,effective", [
    ("openrouter_credits", "code_review_provider_credit", False, 402),
    ("openrouter_key_limit", "code_review_provider_credit", False, 402),
    ("openrouter_in_flight_budget", "code_review_provider_rate_limited", True, 429),
])
def test_review_retains_only_observed_limit_source_and_statuses(source, code, retryable, effective):
    body = budget_error(source)
    body["error"]["message"] = "private provider prose"
    body["error"]["metadata"]["private_detail"] = "other-private-api-key"
    transport = ReviewTransport(lambda _parameters: {"status": 402, "body": body})
    reviewer, row, store, _objects = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["provider_limit_source"] == source
    assert document["provider_raw_http_status"] == 402
    assert document["provider_effective_http_status"] == effective
    assert document["provider_http_status"] == 402
    assert document["error_code"] == code
    assert document["retryable"] is retryable
    assert store.finish_calls[0][5] == 0
    assert len(transport.sent) == 1
    assert "private provider prose" not in json.dumps(document)
    assert "other-private-api-key" not in json.dumps(document)
    assert MINER_KEY not in json.dumps(document)


@pytest.mark.parametrize("error", [
    None, {}, {"code": "402", "metadata": {"limit_source": "openrouter_credits"}},
    {"code": True, "metadata": {"limit_source": "openrouter_credits"}},
    {"code": 402}, {"code": 402, "metadata": []},
    {"code": 402, "metadata": {"limit_source": []}},
    {"code": 402, "metadata": {"limit_source": "private provider prose"}},
    {"code": 500, "metadata": {"limit_source": "openrouter_credits"}},
])
def test_missing_or_malformed_limit_metadata_remains_unknown(error):
    observation = code_review_policy.provider_response_observation(
        {"error": error}, raw_http_status=402, effective_http_status=402,
    )
    assert observation == {
        "provider_limit_source": "unknown", "provider_raw_http_status": 402,
        "provider_effective_http_status": 402,
    }


def test_embedded_error_records_wire_and_effective_status_without_changing_charge():
    transport = ReviewTransport(lambda _parameters: {"status": 200, "body": budget_error()})
    reviewer, row, store, _objects = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["provider_raw_http_status"] == 200
    assert document["provider_effective_http_status"] == 402
    assert document["provider_limit_source"] == "openrouter_in_flight_budget"
    assert document["error_code"] == "code_review_provider_credit"
    assert document["retryable"] is False
    assert store.finish_calls[0][5] is None


def test_credential_echo_never_creates_provider_observations():
    body = budget_error()
    body["error"]["message"] = MINER_KEY
    reviewer, row, store, _objects = _reviewer(
        _archive(_source()), ReviewTransport(lambda _parameters: {"status": 402, "body": body}),
    )
    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["error_code"] == "code_review_credential_echo"
    assert not any(key.startswith("provider_") for key in document)
    assert MINER_KEY not in json.dumps(document)
