"""Source-local provenance never comes from a generic authorization envelope."""
import copy
import json

import pytest

from lab_arena import broker as br, operations
from tests.lab_arena.firecrawl_blocked_site_refusal_test import blocked_site_body, CAPTURED_URL
from tests.lab_arena.firecrawl_unsupported_source_test import routed_call
from tests.lab_arena.firecrawl_tls_source_refusal_test import _routed_call
from tests.lab_arena.generic_http_free_refusal_test import refusal, PARAMETERS
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
