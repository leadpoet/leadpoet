"""Server-owned source headers preserve frozen admission and keyed wire."""

import json
from dataclasses import replace

import pytest

from lab_arena import broker as br, contracts, operations, scoring_provider_compat as compat
from lab_arena import training_capture as capture
from lab_arena.service import LocalObjectStore
from tests.lab_arena.firecrawl_content_fallback_test import (
    HTML, URL, _broker, _envelope, _execute, _generic, REDACTED,
)
from tests.lab_arena.test_lab_arena_broker import FakeLedgerStore, FakeTransport
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.score_compat_capability_test import firecrawl_catalog
from tests.lab_arena.firecrawl_diagnostic_adapter_test import _score_context


PROFILE = {"User-Agent": "Leadpoet-Arena/1.0",
           "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"}
V1 = "miner-score-deepline-compat:v1"


def _catalog(*, headers_supported=True):
    schema = {"type": "object", "properties": {
        "url": {"type": "string"}, "method": {"enum": ["GET"]},
        "follow_redirects": {"type": "boolean"}, "timeout_ms": {"type": "integer"},
        "headers": {"type": "object", "additionalProperties": {"type": "string"}},
    }, "required": ["url"], "additionalProperties": False}
    if not headers_supported:
        del schema["properties"]["headers"]
    free = {"unit": "call", "usdPerUnit": 0, "creditsPerUnit": 0, "currency": "USD"}
    return frozen(
        row("firecrawl_scrape", provider="firecrawl", categories=["research"],
            inputSchema=firecrawl_catalog()["tools"][0]["input_schema"],
            pricing={**free, "usdPerUnit": 0.002, "creditsPerUnit": 0.02}),
        row("generic_http_request", provider="generic_http", categories=["research"],
            inputSchema=schema, pricing=free),
    )


def test_frozen_source_fallback_sends_profile_and_captures_exact_wire(tmp_path):
    class HeaderSensitiveTransport(FakeTransport):
        def send(self, **request):
            response = super().send(**request)
            wire = json.loads(request["body"] or b"{}")
            if wire.get("operation") == "generic_http_request" and wire["payload"].get("headers") != PROFILE:
                return br.ProviderResponse(403, {}, json.dumps({
                    "error": "Source refused an undeclared page-fetch request.",
                    "billing": {"credits_charged": 0, "cost_usd": 0},
                }).encode())
            return response

    objects = LocalObjectStore(tmp_path)
    transport = HeaderSensitiveTransport([(200, _envelope(REDACTED)), (200, _generic())])
    provider, ledger, _ = _broker([], transport=transport,
                                 capture_request=capture.RequestCapture(objects).prepare)
    context = replace(_score_context(), deepline_catalog=_catalog())
    result = provider.execute(context, operation_id="scrapingdog.scrape", parameters={"url": URL},
                              action_sequence=0, timeout_ms=30_000)
    assert result.status == 200 and result.body == HTML.encode()
    first, fallback = list(ledger.calls.values())
    wire = json.loads(transport.sent[1]["body"])
    assert wire["payload"]["headers"] == PROFILE
    assert wire["payload"]["url"] == URL and wire["payload"]["method"] == "GET"
    assert 0 < wire["payload"]["timeout_ms"] <= 30_000
    assert "headers" not in json.loads(transport.sent[0]["body"])["payload"]
    restored = capture.read_request(objects, fallback["call_doc"]["request_capture"])
    assert restored["outbound"]["body"] == wire
    assert restored["outbound"]["body_sha256"] == contracts.hash_bytes(transport.sent[1]["body"])
    assert restored["effective_parameters"]["payload"] == wire["payload"]
    assert restored["hashed_parameters"] == {"url": URL, "dynamic": False}
    assert restored["request_hash"] == contracts.document_hash({"url": URL, "dynamic": False})
    assert first["call_doc"]["request_hash"] == fallback["call_doc"]["request_hash"]
    assert first["call_doc"]["compatibility_version"] == V1
    assert fallback["call_doc"]["compatibility_version"] != V1


@pytest.mark.parametrize("headers", [{}, PROFILE, {"Authorization": "not-a-real-secret"}])
def test_caller_cannot_supply_target_headers_even_for_the_fixed_profile(headers):
    provider, ledger, transport = _broker([])
    result = provider.execute(replace(_score_context(), deepline_catalog=_catalog()),
                              operation_id="deepline.execute",
                              parameters={"tool": "generic_http_request", "payload": {
                                  "url": URL, "method": "GET", "headers": headers,
                              }}, action_sequence=0, timeout_ms=30_000)
    assert result.status == 400 and ledger.calls == {} and transport.sent == []


def test_profile_is_internal_and_native_generic_requests_keep_their_wire():
    params = {"tool": "generic_http_request", "payload": {"url": URL, "method": "GET"}}
    assert "headers" not in json.loads(operations.build_outbound_request(
        "deepline.execute", params, deepline_catalog=_catalog(),
    ).body)["payload"]
    for supplied in ({**params, "source_html_profile": True},
                     {**params, "payload": {**params["payload"], "headers": {}}}):
        with pytest.raises(operations.OperationRequestError):
            operations.build_outbound_request("deepline.execute", supplied, deepline_catalog=_catalog())
    with pytest.raises(operations.OperationRequestError):
        operations.build_outbound_request("deepline.execute", {
            **params, "payload": {**params["payload"], "headers": {}},
        }, source_html_profile=True)
    with pytest.raises(operations.OperationRequestError):
        operations.build_outbound_request("scrapingdog.scrape", {"url": URL}, source_html_profile=True)


def test_unsupported_frozen_header_schema_keeps_v1_without_orphaning_calls():
    provider, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, _generic())])
    context = replace(_score_context(), deepline_catalog=_catalog(headers_supported=False))
    for _ in range(2):
        result = provider.execute(context, operation_id="scrapingdog.scrape", parameters={"url": URL},
                                  action_sequence=0, timeout_ms=30_000)
        assert result.status == 200
    assert len(transport.sent) == 2
    fallback = next(v for v in ledger.calls.values() if v["call_doc"].get("provider_attempt") == 2)
    assert fallback["call_doc"]["compatibility_version"] == V1
    assert "headers" not in json.loads(transport.sent[1]["body"])["payload"]
    assert all(v["kind"] == "settlement" for v in ledger.calls.values())


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("state", ["reservation", "dispatch", "settlement"])
def test_restart_reproduces_legacy_and_profile_wire_without_capture(monkeypatch, legacy, state):
    class RecoveryStore(FakeLedgerStore):
        def recover_deepline_response(self, **kwargs):
            return self.settle_call(**{key: kwargs[key] for key in (
                "run_id", "lease_token_hash", "call_identity", "actual_microusd",
                "terminal_response", "lease_ttl_seconds",
            )})

    original_route_for = compat.route_for
    if legacy:
        def legacy_route(**kwargs):
            route = original_route_for(**kwargs)
            return replace(route, compatibility_version=V1) if route else None
        monkeypatch.setattr(compat, "route_for", legacy_route)
    provider, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, _generic())],
                                        store=RecoveryStore())
    assert _execute(provider).status == 200
    fallback = next(v for v in ledger.calls.values() if v["call_doc"].get("provider_attempt") == 2)
    assert "request_capture" not in fallback["call_doc"]
    original_body = transport.sent[1]["body"]
    assert ("headers" not in json.loads(original_body)["payload"]) is legacy
    if state != "settlement":
        fallback["kind"] = state
        fallback.pop("terminal")
    monkeypatch.setattr(compat, "route_for", original_route_for)

    class ResumeTransport(FakeTransport):
        def send(self, **request):
            if request["method"] == "GET":
                self.sent.append(request)
                return br.ProviderResponse(200, {"x-deepline-idempotency-supported": "true"}, json.dumps({
                    "requestId": _generic()["job_id"], "toolId": "generic_http_request",
                    "executionRecovery": {"idempotencyKey": fallback["call_doc"]["deepline_execution_key"], "state": "running"},
                }).encode())
            assert request["body"] == original_body
            return super().send(**request)

    resumed = ResumeTransport([(200, _generic())])
    restarted, _, _ = _broker([], store=ledger, transport=resumed)
    assert _execute(restarted).status == 200
    assert len(resumed.sent) == {"reservation": 1, "dispatch": 2, "settlement": 0}[state]


@pytest.mark.parametrize("version", [None, False, 1, "unknown-profile-version"])
def test_unknown_saved_source_wire_version_never_dispatches(version):
    provider, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, _generic())])
    assert _execute(provider).status == 200
    fallback = next(v for v in ledger.calls.values() if v["call_doc"].get("provider_attempt") == 2)
    fallback["call_doc"]["compatibility_version"] = version
    assert _execute(provider).call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == 2


@pytest.mark.parametrize("headers_supported", [False, True])
def test_old_v1_reserved_wire_repairs_only_its_existing_capture_pointer(monkeypatch, tmp_path, headers_supported):
    objects = LocalObjectStore(tmp_path)
    original_route_for = compat.route_for

    def legacy_route(**kwargs):
        route = original_route_for(**kwargs)
        return replace(route, compatibility_version=V1) if route else None

    monkeypatch.setattr(compat, "route_for", legacy_route)
    context = replace(_score_context(), deepline_catalog=_catalog(headers_supported=headers_supported))
    provider, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, _generic())],
                                         capture_request=capture.RequestCapture(objects).prepare)

    def execute(broker):
        return broker.execute(context, operation_id="scrapingdog.scrape", parameters={"url": URL},
                              action_sequence=0, timeout_ms=30_000)

    assert execute(provider).status == 200
    fallback = next(v for v in ledger.calls.values() if v["call_doc"].get("provider_attempt") == 2)
    pointer = dict(fallback["call_doc"]["request_capture"])
    original_capture = objects.get(pointer["ref"])
    original_body = transport.sent[1]["body"]
    objects._path(pointer["ref"]).unlink()  # Only this test's temporary blob.
    fallback["kind"] = "reservation"
    fallback.pop("terminal")
    monkeypatch.setattr(compat, "route_for", original_route_for)
    restarted, _, resumed = _broker([(200, _generic())], store=ledger,
                                    capture_request=capture.RequestCapture(objects).prepare)
    assert execute(restarted).status == 200
    assert resumed.sent[0]["body"] == original_body
    assert fallback["call_doc"]["request_capture"] == pointer
    assert objects.get(pointer["ref"]) == original_capture
    restored = capture.read_request(objects, pointer)
    assert restored["outbound"]["body"] == json.loads(original_body)
    assert "headers" not in restored["effective_parameters"]["payload"]
