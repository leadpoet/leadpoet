"""Source-local provenance never comes from a generic authorization envelope."""
import copy
import json

import pytest

from lab_arena import broker as br, operations
from tests.lab_arena.firecrawl_blocked_site_refusal_test import blocked_site_body, CAPTURED_URL
from tests.lab_arena.firecrawl_unsupported_source_test import routed_call
from tests.lab_arena.firecrawl_tls_source_refusal_test import _routed_call
from tests.lab_arena.generic_http_free_refusal_test import refusal, PARAMETERS, Transport, catalog
from tests.lab_arena.test_lab_arena_broker import deepline_generic_http_denial, FakeLedgerStore


@pytest.mark.parametrize("kind", ["unsupported", "tls"])
def test_known_billed_source_refusal_marker_is_durable_and_replays_exactly(kind):
    result, broker, store, transport, context, arguments = (
        routed_call(url=CAPTURED_URL, body=blocked_site_body()) if kind == "unsupported" else _routed_call()
    )
    assert result.status == 403 and result.call["error_code"] == "provider_request_refused"
    assert result.headers[br.SOURCE_UNAVAILABLE_HEADER] == br.SOURCE_UNAVAILABLE_VALUE
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert terminal["headers"][br.SOURCE_UNAVAILABLE_HEADER] == br.SOURCE_UNAVAILABLE_VALUE
    assert terminal["call_succeeded"] is False
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.body == result.body and replay.headers == result.headers
    assert replay.call["actual_microusd"] == result.call["actual_microusd"] == 0
    assert len(transport.sent) == before


def test_unknown_bill_and_failed_settlement_never_deliver_source_proof(monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    result, *_ = _routed_call(credits=None)
    assert br.SOURCE_UNAVAILABLE_HEADER not in result.headers

    class FailedSettlement(FakeLedgerStore):
        def settle_call(self, **_kwargs):
            raise RuntimeError("offline settlement failure")

    failed, *_ = _routed_call(store=FailedSettlement())
    assert br.SOURCE_UNAVAILABLE_HEADER not in failed.headers


@pytest.mark.parametrize("owner,source,category", [
    ("workspace", "managed", "authorization"),
    ("workspace", "env", "provider_auth"),
    ("deepline_managed", "env", "authorization"),
    ("deepline_managed", "env", "provider_auth"),
])
def test_ambiguous_or_account_generic_http_denial_never_gets_source_proof(owner, source, category):
    body = deepline_generic_http_denial()
    body.update(credential_owner=owner, credential_source=source, error_category=category)
    response = br.ProviderResponse(403, {}, json.dumps(body).encode())
    assert not br._proved_source_unavailable("deepline", PARAMETERS, response)


@pytest.mark.parametrize("status", [401, 403])
def test_generic_http_source_proof_requires_credential_free_nonretryable_receipt(status):
    body = refusal(status)
    assert br._proved_source_unavailable("deepline", PARAMETERS,
        br.ProviderResponse(status, {}, json.dumps(body).encode()))
    for change in ("owner", "source", "retryable", "origin", "category"):
        changed = copy.deepcopy(body)
        if change == "owner": changed["credential_owner"] = "workspace"
        elif change == "source": changed["credential_source"] = "managed"
        elif change == "retryable": changed["tool_error"]["retryable"] = True
        elif change == "origin": changed["failure_origin"] = "provider_account"
        else: changed["error_category"] = "provider_authentication"
        assert not br._proved_source_unavailable("deepline", PARAMETERS,
            br.ProviderResponse(status, {}, json.dumps(changed).encode()))


def test_provider_cannot_supply_the_host_marker():
    _, headers, _ = operations.sanitize_response("deepline.execute", 200,
        {br.SOURCE_UNAVAILABLE_HEADER: br.SOURCE_UNAVAILABLE_VALUE}, b'{"result":[]}')
    assert br.SOURCE_UNAVAILABLE_HEADER not in headers


def test_legacy_canonical_reply_is_not_retroactively_given_source_authority():
    result, broker, store, transport, context, arguments = routed_call(
        url=CAPTURED_URL, body=blocked_site_body())
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    terminal["headers"].pop(br.SOURCE_UNAVAILABLE_HEADER)
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.body == result.body
    assert br.SOURCE_UNAVAILABLE_HEADER not in replay.headers
    assert len(transport.sent) == before


@pytest.mark.parametrize('field,value', [
    ('upstream_status', 401), ('upstream_status', True),
    ('tool_error.code', 'PROVIDER_AUTHORIZATION_FAILED'),
    ('tool_error.provider', 'different'), ('tool_error.operation', 'different'),
    ('tool_error.statusCode', 401), ('tool_error.statusCode', True),
])
def test_legacy_linkedin_shape_cannot_bypass_stronger_consistent_source_proof(field, value):
    body = blocked_site_body()
    message = ('Firecrawl does not support scraping LinkedIn URLs. Use a native '
               'HarvestAPI LinkedIn operation when it covers the requested shape.')
    body.update(error=message, detail=message, message=message)
    target = body
    parts = field.split('.')
    if len(parts) == 2: target = body[parts[0]]
    target[parts[-1]] = value
    parameters = {'tool': 'firecrawl_scrape', 'payload': {'url': 'https://www.linkedin.com/company/example/'}}
    response = br.ProviderResponse(422, {}, json.dumps(body).encode())
    assert br._deepline_firecrawl_unsupported_source_refusal(parameters, response)
    assert not br._proved_source_unavailable('deepline', parameters, response)


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("category", ["provider_auth", "authentication"])
def test_generic_http_typed_authentication_category_is_source_local(status, category):
    # Retained November 17 generic_http_request responses use the top-level
    # provider_auth category with tool_error.category=authentication.
    body = refusal(status)
    body["tool_error"]["category"] = category
    assert br._proved_source_unavailable("deepline", PARAMETERS,
        br.ProviderResponse(status, {}, json.dumps(body).encode()))


@pytest.mark.parametrize("field,value", [
    ("credential_owner", None), ("credential_owner", "miner"),
    ("credential_owner", "workspace"), ("credential_owner", "deepline_managed"),
    ("credential_source", None), ("credential_source", "env"),
    ("failure_origin", "caller"), ("failure_origin", "provider_account"),
    ("error_category", "authentication"), ("error_category", None),
    ("upstream_status", 401), ("upstream_status", True),
    ("provider", "scrapingdog"), ("operation", "another_operation"),
    ("tool_error", None), ("tool_error.code", "different"),
    ("tool_error.provider", "different"), ("tool_error.operation", "different"),
    ("tool_error.statusCode", 401), ("tool_error.statusCode", True),
    ("tool_error.category", "authorization"), ("tool_error.category", None),
    ("tool_error.origin", "caller"), ("tool_error.origin", None),
    ("tool_error.retryable", True), ("tool_error.retryable", None),
])
def test_typed_authentication_does_not_relax_source_provenance(field, value):
    body = refusal(403)
    body["tool_error"]["category"] = "authentication"
    target = body
    parts = field.split(".")
    if len(parts) == 2:
        target = body[parts[0]]
    if value is None:
        target.pop(parts[-1])
    else:
        target[parts[-1]] = value
    assert not br._proved_source_unavailable("deepline", PARAMETERS,
        br.ProviderResponse(403, {}, json.dumps(body).encode()))


@pytest.mark.parametrize("status", [401, 402, 403])
@pytest.mark.parametrize("owner", ["miner", "workspace", "deepline_managed"])
def test_account_authentication_cannot_become_source_unavailable(status, owner):
    body = refusal(status)
    body.update(credential_owner=owner, credential_source="env")
    body["tool_error"]["category"] = "authentication"
    assert not br._proved_source_unavailable("deepline", PARAMETERS,
        br.ProviderResponse(status, {}, json.dumps(body).encode()))


@pytest.mark.parametrize("status,body,provenance", [
    (403, b"Forbidden", None), (403, b'{"error":"authentication"}', None),
    (402, None, None), (403, None, "response_too_large"),
])
def test_bare_or_synthetic_authentication_never_proves_source_refusal(status, body, provenance):
    document = refusal(status)
    document["tool_error"]["category"] = "authentication"
    assert not br._proved_source_unavailable("deepline", PARAMETERS,
        br.ProviderResponse(status, {}, body or json.dumps(document).encode(), provenance))


def test_retryable_firecrawl_site_error_remains_unavailable():
    from tests.lab_arena.firecrawl_tls_source_refusal_test import CAPTURED

    body = copy.deepcopy(CAPTURED)
    body["upstream_error_code"] = "SCRAPE_SITE_ERROR"
    body["tool_error"]["retryable"] = True
    parameters = {"tool": "firecrawl_scrape", "payload": {"url": "https://example.com/"}}
    response = br.ProviderResponse(502, {}, json.dumps(body).encode())
    assert not br._provider_request_refused("deepline", parameters, response)
    assert not br._proved_source_unavailable("deepline", parameters, response)


def typed_authentication_broker_result(funding="miner_key"):
    from dataclasses import replace
    from tests.lab_arena.test_lab_arena_broker import CONTEXT, DL_KEY, make_broker

    body = refusal(403)
    body["tool_error"]["category"] = "authentication"
    transport = Transport(body, 403)
    broker, store, _ = make_broker(transport=transport, credential_for=lambda *_: DL_KEY,
        provider_funding_source_for=lambda *_: funding)
    context = replace(CONTEXT, deepline_catalog=catalog())
    arguments = dict(operation_id="deepline.execute", parameters=PARAMETERS,
                     action_sequence=0, timeout_ms=1000)
    result = broker.execute(context, **arguments)
    return result, broker, store, transport, context, arguments


@pytest.mark.parametrize("funding", ["host", "miner_key"])
def test_typed_authentication_marker_is_settled_and_replayed_without_paid_retry(funding):
    result, broker, store, transport, context, arguments = typed_authentication_broker_result(funding)
    assert result.status == 403
    assert result.headers[br.SOURCE_UNAVAILABLE_HEADER] == br.SOURCE_UNAVAILABLE_VALUE
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert terminal["headers"] == result.headers
    assert terminal["call_succeeded"] is False
    replay = broker.execute(context, **arguments)
    assert replay.body == result.body and replay.headers == result.headers
    assert replay.call["actual_microusd"] == result.call["actual_microusd"] == 0
    assert result.call["funding_source"] == funding
    assert len(transport.sent) == store.log.count("settle") == 1
