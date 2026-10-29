"""Structured source JSON keeps its source proof, whole body, and paid identity."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import operations, scoring_provider_compat as compat, shim
from tests.lab_arena.compatibility_failure_diagnostic_test import _event
from tests.lab_arena.firecrawl_content_fallback_test import _broker, _execute
from tests.lab_arena.firecrawl_diagnostic_adapter_test import URL, _envelope, _route, _score_context
from tests.lab_arena.test_lab_arena_broker import DL_KEY


VALUE = {"url": "https://publisher.example/article", "archived_snapshots": {}}
FINAL = "https://example.com/about/data"


def envelope(value=VALUE, *, media="application/json; charset=utf-8", status=200):
    document = _envelope(value, status=status)
    document["result"]["data"]["metadata"].update(contentType=media, url=FINAL)
    return document


def adapt(document):
    return compat.adapt_response_with_trusted_url(
        _route(), status=200, headers={"set-cookie": "private=test"},
        body=json.dumps(document).encode(),
    )


@pytest.mark.parametrize("value", [VALUE, {"label": "café", "items": [False, None]}, [1, {}], {}, []])
def test_structured_source_json_remains_canonical_json_after_source_proof(value):
    assert adapt(envelope(value)) == (
        200, {"content-type": "application/json; charset=utf-8"},
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(), FINAL,
    )


def test_json_success_keeps_exact_charge_identity_trajectory_and_restart_replay():
    broker, ledger, transport = _broker([(200, envelope())])
    result = _execute(broker)
    assert result.status == 200 and json.loads(result.body) == VALUE
    assert result.headers["content-type"] == "application/json"
    assert result.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == FINAL
    assert result.call["actual_microusd"] == 2000
    assert "compatibility_failure_code" not in _event(result)["call"]
    assert _event(result)["http_status"] == 200
    saved = ledger.calls[result.call["call_identity"]]
    assert saved["actual"] == 2000 and saved["terminal"]["call_succeeded"] is True
    assert saved["call_doc"]["compatibility_version"] == "miner-score-deepline-compat:v1"
    assert saved["call_doc"]["adapter"] == "firecrawl_raw_html"
    request = json.loads(transport.sent[0]["body"])
    assert request["operation"] == "firecrawl_scrape"
    assert request["payload"]["url"] == URL
    assert request["payload"]["formats"] == ["rawHtml"]
    assert "x-deepline-execute-response-contract" not in transport.sent[0]["headers"]
    restarted, _, replay_transport = _broker([], store=ledger)
    for _ in range(2):
        replay = _execute(restarted)
        assert replay.status == result.status and replay.body == result.body
        assert replay.headers == result.headers
        assert replay.call["call_identity"] == result.call["call_identity"]
        assert replay.call["idempotent"] is True
    assert replay_transport.sent == []
    assert len(transport.sent) == ledger.log.count("settle") == len(ledger.calls) == 1


@pytest.mark.parametrize("field,value", [
    ("sourceURL", "https://wrong.example/"), ("url", "https://127.0.0.1/"),
    ("url", "http://example.com/"), ("url", "https://private.local/"),
    ("statusCode", True), ("statusCode", "200"), ("statusCode", None),
    ("contentType", None), ("contentType", "text/html"),
    ("contentType", "application/pdf"), ("contentType", "application/problem+json"),
    ("contentType", ["application/json"]),
])
def test_json_never_relaxes_metadata_or_media_proof(field, value):
    document = envelope()
    document["result"]["data"]["metadata"][field] = value
    broker, ledger, transport = _broker([(200, document)])
    result = _execute(broker)
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert operations.TRUSTED_RESPONSE_URL_HEADER not in result.headers
    assert _event(result)["http_status"] == 502
    assert len(transport.sent) == ledger.log.count("settle") == 1
    assert ledger.calls[result.call["call_identity"]]["actual"] == 2000


@pytest.mark.parametrize("status", [100, 199, 301, 304, 400, 401, 403, 404, 410, 429, 500, 503, 599])
def test_non_success_json_never_exposes_error_content_or_opens_rescue(status):
    broker, ledger, transport = _broker([(200, envelope(status=status))])
    result = _execute(broker)
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert operations.TRUSTED_RESPONSE_URL_HEADER not in result.headers
    assert _event(result)["http_status"] == 502
    assert len(transport.sent) == ledger.log.count("settle") == len(ledger.calls) == 1


@pytest.mark.parametrize("value", [True, 1, 1.5, None])
def test_json_scalars_remain_invalid(value):
    with pytest.raises(compat.CompatibilityResponseError, match="missing_firecrawl_raw_html"):
        adapt(envelope(value))


@pytest.mark.parametrize("text", ['{"available":true}', "[REDACTED]", "%PDF-fixture", "JVBERi0fixture"])
def test_existing_declared_json_strings_keep_the_existing_html_path(text):
    document = envelope(text)
    if text in {"[REDACTED]", "%PDF-fixture", "JVBERi0fixture"}:
        with pytest.raises(compat.CompatibilityResponseError):
            adapt(document)
    else:
        assert adapt(document) == (200, {"content-type": "text/html; charset=utf-8"}, text.encode(), FINAL)


@pytest.mark.parametrize("value", [{"nonfinite": float("nan")}, {"oversized": "x" * 2_097_153}])
def test_invalid_or_oversized_json_settles_once_without_partial_content(value):
    broker, ledger, transport = _broker([(200, envelope(value))])
    result = _execute(broker)
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert DL_KEY.encode() not in result.body
    assert _event(result)["http_status"] == 502
    assert _event(result)["call"]["actual_microusd"] == 2000
    replay = _execute(broker)
    assert replay.body == result.body and replay.call["idempotent"] is True
    assert len(transport.sent) == ledger.log.count("settle") == len(ledger.calls) == 1
    assert ledger.calls[result.call["call_identity"]]["actual"] == 2000


def test_json_keeps_existing_credential_refusal_and_uncertain_cost_state():
    broker, ledger, transport = _broker([(200, envelope({"credential": DL_KEY}))])
    result = _execute(broker)
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert DL_KEY.encode() not in result.body
    assert result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    assert DL_KEY not in repr(_event(result))
    assert sum(request["method"] == "POST" for request in transport.sent) == len(ledger.calls) == 1
    assert ledger.log == ["reserve", "dispatch", "uncertain"]


def test_whole_json_at_exact_byte_cap_is_preserved_and_one_byte_over_fails():
    limit = operations.OPERATIONS["scrapingdog.scrape"].max_response_bytes
    overhead = len(b'{"text":""}')
    value = {"text": "x" * (limit - overhead)}
    assert len(adapt(envelope(value))[2]) == limit
    value["text"] += "x"
    with pytest.raises(compat.CompatibilityResponseError, match="firecrawl_json_body_too_large"):
        adapt(envelope(value))


@pytest.mark.parametrize("damage", ["incomplete", "missing_result", "list_data"])
def test_malformed_native_envelopes_remain_invalid(damage):
    document = envelope()
    if damage == "incomplete":
        document["status"] = "running"
    elif damage == "missing_result":
        document.pop("result")
    else:
        document["result"]["data"] = [VALUE]
    with pytest.raises(compat.CompatibilityResponseError):
        adapt(document)


@pytest.mark.parametrize("level", ["document", "result", "data"])
@pytest.mark.parametrize("field,value", [
    ("error", {}), ("tool_error", "failure"), ("success", False), ("isError", True),
    ("status", "failed"), ("status", True),
])
def test_native_failure_flags_cannot_supply_structured_json(level, field, value):
    document = envelope()
    node = {"document": document, "result": document["result"], "data": document["result"]["data"]}[level]
    node[field] = value
    with pytest.raises(compat.CompatibilityResponseError):
        adapt(document)


def test_source_json_keys_are_not_native_provider_flags():
    value = {"error": "public API result", "success": False, "status": 404, "isError": True}
    assert json.loads(adapt(envelope(value))[2]) == value


@pytest.mark.parametrize("http_status", [401, 402, 403, 429, 500, 503])
def test_outer_account_and_transport_failures_cannot_supply_source_json(http_status):
    with pytest.raises(compat.CompatibilityResponseError, match="deepline_http_error"):
        compat.adapt_response_with_trusted_url(
            _route(), status=http_status, headers={}, body=json.dumps(envelope()).encode(),
        )


@pytest.mark.parametrize("has_snapshot", [False, True])
def test_unchanged_wayback_caller_uses_primary_json_without_a_rescue(monkeypatch, has_snapshot):
    from qualification.scoring import intent_verification_three_stage as intent

    original = VALUE["url"]
    availability_url = "https://archive.org/wayback/available?url=" + original
    wire_url = shim._trusted_page_fetch("GET", shim.strip_caller_credentials(availability_url, {})[0], b"")[1]["url"]
    snapshot = "https://web.archive.org/web/20261023010203/" + original
    value = {"url": original, "archived_snapshots": {}}
    if has_snapshot:
        value["archived_snapshots"]["closest"] = {
            "available": True, "status": "200", "timestamp": "20261023010203", "url": snapshot,
        }
    primary = envelope(value)
    primary["result"]["data"]["metadata"].update(sourceURL=wire_url, url=wire_url)
    html = '<html><body><a href="' + original + '">Original</a>' + "Archived fixture. " * 40 + "</body></html>"
    snapshot_wire = shim._trusted_page_fetch("GET", snapshot, b"")[1]["url"]
    archived = _envelope(html, source_url=snapshot_wire)
    archived["result"]["data"]["metadata"]["url"] = snapshot_wire
    responses = [(200, primary)] + ([(200, archived)] if has_snapshot else [])
    broker, ledger, transport = _broker(responses)
    dispatched = []

    def dispatch(operation_id, parameters, timeout_ms):
        dispatched.append(parameters["url"])
        result = broker.execute(_score_context(), operation_id=operation_id, parameters=parameters,
                                action_sequence=len(dispatched), timeout_ms=timeout_ms)
        assert _event(result)["http_status"] == 200
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
            return send(self, httpx.Request("GET", url))

    monkeypatch.setattr(intent, "httpx", SimpleNamespace(AsyncClient=Client))
    result = asyncio.run(intent._try_wayback(original))
    if has_snapshot:
        assert result["ok"] is True and result["content"] == html
        assert result["meta"]["ownership_archive_url"] == snapshot
        assert dispatched == [wire_url, snapshot_wire]
    else:
        assert result["ok"] is False and result["stage"] == "wayback_no_snapshot"
        assert dispatched == [wire_url]
    assert len(transport.sent) == ledger.log.count("settle") == len(dispatched)
    assert all(call["actual"] == 2000 for call in ledger.calls.values())
