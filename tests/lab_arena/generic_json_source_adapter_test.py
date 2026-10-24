"""Bound public JSON remains JSON through the existing source rescue."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import operations, scoring_provider_compat as compat, shim
from tests.lab_arena.generic_html_source_adapter_test import FINAL, adapt, envelope, route
from tests.lab_arena.firecrawl_content_fallback_test import _broker, _execute, _generic
from tests.lab_arena.firecrawl_diagnostic_adapter_test import REDACTED, _envelope, _score_context
from tests.lab_arena.test_lab_arena_broker import DL_KEY


def json_envelope(value=None):
    document = envelope()
    document["toolResponse"]["rawV2"].update(
        headers={"content-type": "application/json"},
        data={"available": True, "count": 2} if value is None else value,
    )
    return document


@pytest.mark.parametrize("value", [{"available": True, "label": "café"}, [1, {"found": False}], {}])
def test_authenticated_structured_json_survives_as_json_without_provider_headers(value):
    document = json_envelope(value)
    document["toolResponse"]["rawV2"]["headers"]["set-cookie"] = "private=test"
    status, headers, body, final_url = adapt(document)
    assert status == 200 and final_url == FINAL
    assert headers == {"content-type": "application/json; charset=utf-8"}
    assert json.loads(body) == value


@pytest.mark.parametrize("field,value", [
    ("provider", "firecrawl"), ("operation", "another_request"), ("method", "POST"),
    ("requested_url", FINAL), ("final_url", "https://127.0.0.1/"),
    ("status_code", True), ("status_code", "200"), ("ok", False),
    ("headers", {"content-type": "application/json", "Content-Type": "text/html"}),
    ("headers", {"content-type": "application/pdf"}),
    ("data", '{"incomplete":'), ("data", "[REDACTED]"),
    ("data", 1), ("data", True), ("data", None),
    ("data", {"nonfinite": float("nan")}),
])
def test_json_never_relaxes_native_binding_status_header_or_data_proof(field, value):
    document = json_envelope()
    document["toolResponse"]["rawV2"][field] = value
    with pytest.raises(compat.CompatibilityResponseError):
        adapt(document)


@pytest.mark.parametrize("field", ["error", "tool_error"])
def test_native_error_envelope_cannot_supply_json(field):
    document = json_envelope()
    document[field] = {"code": "provider_failure"}
    with pytest.raises(compat.CompatibilityResponseError):
        adapt(document)


def test_malformed_native_json_is_not_accepted():
    with pytest.raises(compat.CompatibilityResponseError, match="invalid_generic_html_envelope"):
        compat.adapt_response_with_trusted_url(route(), status=200, headers={}, body=b'{"status":')


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_wayback_caller_routes_both_legs_with_source_bound_snapshot_https(monkeypatch, scheme):
    from qualification.scoring import intent_verification_three_stage as intent

    original = "https://publisher.example/article"
    availability_url = "https://archive.org/wayback/available?url=" + original
    snapshot = scheme + "://web.archive.org/web/20261023010203/" + original
    availability = {"url": original, "archived_snapshots": {"closest": {
        "available": True, "status": "200", "timestamp": "20261023010203", "url": snapshot,
    }}}
    availability_wire = shim._trusted_page_fetch("GET", shim.strip_caller_credentials(availability_url, {})[0], b"")[1]["url"]
    native = _envelope(REDACTED, source_url=availability_wire)
    native["result"]["data"]["metadata"]["url"] = availability_wire
    generic = json_envelope(availability)
    generic["toolResponse"]["rawV2"].update(requested_url=availability_wire, final_url=availability_wire)
    # Only synthetic archived HTML is used. This proves route/source binding,
    # not that the captured live snapshot contains usable evidence.
    html = '<html><body><a href="' + original + '">Original</a>' + "Archived fixture. " * 40 + "</body></html>"
    transport_snapshot = "https" + snapshot[4:] if scheme == "http" else snapshot
    snapshot_wire = shim._trusted_page_fetch("GET", transport_snapshot, b"")[1]["url"]
    archived = _envelope(html, source_url=snapshot_wire)
    archived["result"]["data"]["metadata"]["url"] = snapshot_wire
    broker, ledger, transport = _broker([(200, native), (200, generic), (200, archived)])
    calls = []
    dispatched = []

    def dispatch(operation_id, parameters, timeout_ms):
        dispatched.append(parameters["url"])
        result = broker.execute(_score_context(), operation_id=operation_id, parameters=parameters,
                                action_sequence=len(dispatched), timeout_ms=timeout_ms)
        return result.status, result.headers, result.body, result.headers.get(operations.TRUSTED_RESPONSE_URL_HEADER, "")

    monkeypatch.setattr(shim, "trusted_scorer_mode", lambda: True)
    monkeypatch.setattr(shim, "_input_deepline_catalog", lambda: None)
    monkeypatch.setattr(shim, "_dispatch_with_response_url", dispatch)
    send, _ = shim._httpx_send(httpx)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            calls.append(url)
            return send(self, httpx.Request("GET", url))

    monkeypatch.setattr(intent, "httpx", SimpleNamespace(AsyncClient=Client))
    result = asyncio.run(intent._try_wayback(original))
    assert calls == [availability_url, transport_snapshot]
    assert dispatched == [availability_wire, snapshot_wire]
    assert len(transport.sent) == 3
    assert result["ok"] is True and result["content"] == html
    assert result["meta"]["ownership_archive_url"] == transport_snapshot
    assert [call["actual"] for call in ledger.calls.values()] == [2000, 0, 2000]


def generic_json(value):
    document = _generic()
    document["toolResponse"]["rawV2"].update(headers={"content-type": "application/json"}, data=value)
    return document


def test_json_rescue_retains_exact_charges_and_replays_without_second_dispatch():
    value = {"available": True, "items": ["one"]}
    broker, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, generic_json(value))])
    result = _execute(broker)
    assert result.status == 200 and json.loads(result.body) == value
    assert result.headers["content-type"] == "application/json"
    primary, fallback = list(ledger.calls.values())
    assert [primary["actual"], fallback["actual"]] == [2000, 0]
    assert primary["terminal"]["call_succeeded"] is False
    assert fallback["terminal"]["call_succeeded"] is True
    replay = _execute(broker)
    assert replay.body == result.body and replay.call["idempotent"] is True
    assert len(transport.sent) == ledger.log.count("settle") == 2


@pytest.mark.parametrize("value", [{"credential": DL_KEY}, {"large": "x" * 2_097_153}])
def test_json_rescue_keeps_broker_privacy_and_final_response_size_limits(value):
    broker, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, generic_json(value))])
    result = _execute(broker)
    assert result.status == 502
    assert DL_KEY.encode() not in result.body
    assert result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert sum(request["method"] == "POST" for request in transport.sent) == 2
