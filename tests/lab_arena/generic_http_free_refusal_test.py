"""A proved free target-page refusal must not block unrelated research."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, deepline_catalog, provider_costs
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, DL_KEY, deepline_generic_http_denial, make_broker,
)

PARAMETERS = {"tool": "generic_http_request", "payload": {"url": "https://example.com/"}}


def refusal(status=401):
    value = deepline_generic_http_denial("public-execute:page-refused")
    value.update(job_id=value["request_id"], status="failed", credential_owner="none",
                 credential_source="none", error_category="provider_auth", upstream_status=status)
    value["tool_error"].update(statusCode=status, category="provider_auth")
    return value


def catalog(billing_source="free"):
    return frozen(row("generic_http_request", provider="generic_http", billingSource=billing_source,
        pricing={"unit": "call", "usdPerUnit": 0, "creditsPerUnit": 0, "currency": "USD", "displayText": "Free"},
        inputSchema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}))


class Transport:
    def __init__(self, document, status=401, *, headers=None, provenance=None):
        self.document, self.status = document, status
        self.headers, self.provenance = headers or {}, provenance
        self.sent = []

    def send(self, **request):
        self.sent.append(request)
        if request["method"] == "POST":
            return br.ProviderResponse(self.status, self.headers, json.dumps(self.document).encode(), self.provenance)
        # Exact readback has no final charge. It is not a zero-cost proof.
        return br.ProviderResponse(200, {}, b'{"recent":{"entries":[]}}')


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("funding", ["host", "miner_key"])
@pytest.mark.parametrize("billing_source", [None, "free"])
def test_free_target_refusal_settles_and_replays_without_billing_lookup(status, funding, billing_source):
    transport = Transport(refusal(status), status)
    broker, store, _ = make_broker(transport=transport, credential_for=lambda *_: DL_KEY,
        provider_funding_source_for=lambda *_: funding)
    context = replace(CONTEXT, deepline_catalog=catalog(billing_source))
    arguments = dict(operation_id="deepline.execute", parameters=PARAMETERS,
                     action_sequence=0, timeout_ms=1000)
    result = broker.execute(context, **arguments)
    assert result.status == 403 and result.body == b'{"error":{"code":"provider_request_refused"}}'
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_frozen_catalog_refusal_zero"
    replay = broker.execute(context, **arguments)
    assert replay.body == result.body and replay.call["idempotent"] is True
    assert replay.call["actual_microusd"] == 0
    assert len(transport.sent) == store.log.count("settle") == 1


@pytest.mark.parametrize("change", ["unknown_label", "positive_usd", "positive_credits", "unknown_usd", "unknown_credits", "unverified"])
def test_older_listing_requires_exact_frozen_free_call_price(change):
    entry = deepcopy(deepline_catalog.tool_entry(catalog(None), "generic_http_request"))
    if change == "unknown_label": entry["pricing"]["displayText"] = "Contact us"
    elif change == "positive_usd": entry["pricing"]["usd_per_unit"] = 0.01
    elif change == "positive_credits": entry["pricing"]["credits_per_unit"] = 1
    elif change == "unknown_usd": entry["pricing"]["usd_per_unit"] = None
    elif change == "unknown_credits": entry["pricing"]["credits_per_unit"] = None
    assert provider_costs.deepline_free_completed_cost(PARAMETERS, 401, refusal(),
        catalog_entry=entry, request_refused=change != "unverified") is None


@pytest.mark.parametrize("change", ["legacy", "paid", "unit", "tool", "unverified", "status", "pending", "job", "billing", "nested_billing", "nested_usage", "deep"])
def test_free_refusal_never_guesses_unknown_or_conflicting_accounting(change):
    entry = deepcopy(deepline_catalog.tool_entry(catalog(), "generic_http_request"))
    document = refusal()
    parameters = deepcopy(PARAMETERS)
    status, verified = 401, True
    if change == "legacy": entry = None
    elif change == "paid": entry["pricing"]["billing_source"] = "deepline_credits"
    elif change == "unit": entry["pricing"]["unit"] = "usage"
    elif change == "tool": parameters["tool"] = "exa_search"
    elif change == "unverified": verified = False
    elif change == "status": status = 502
    elif change == "pending": document["status"] = "running"
    elif change == "job": document["job_id"] = ""
    elif change == "billing": document["billing"] = None
    elif change == "nested_billing": document["tool_error"]["billing"] = {"credits_charged": 1}
    elif change == "nested_usage": document["metadata"] = [{"usage": {"credits": 0}}]
    elif change == "deep":
        nested = document
        for _ in range(13): nested["metadata"] = {}; nested = nested["metadata"]
    assert provider_costs.deepline_free_completed_cost(parameters, status, document,
        catalog_entry=entry, request_refused=verified) is None


@pytest.mark.parametrize("change", ["account", "wrong_request", "wrong_alias", "header", "synthetic", "wrong_tool"])
def test_zero_refusal_requires_bound_real_target_response(monkeypatch, change):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    document, headers, provenance = refusal(), {}, None
    if change == "account": document["credential_owner"] = "workspace"
    elif change == "wrong_request": document["tool_error"]["requestId"] = "different"
    elif change == "wrong_alias": document["request_id"] = "different"
    elif change == "header": headers["x-deepline-request-id"] = "different"
    elif change == "synthetic": provenance = "response_too_large"
    elif change == "wrong_tool": document["operation"] = "exa_search"
    transport = Transport(document, headers=headers, provenance=provenance)
    broker, _, _ = make_broker(transport=transport, credential_for=lambda *_: DL_KEY)
    result = broker.execute(replace(CONTEXT, deepline_catalog=catalog()),
        operation_id="deepline.execute", parameters=PARAMETERS, action_sequence=0, timeout_ms=1000)
    assert result.call.get("cost_basis") != "deepline_frozen_catalog_refusal_zero"
    assert result.call.get("actual_microusd") is None
