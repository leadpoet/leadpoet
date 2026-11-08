"""Observed bytes from failed streams are private evidence, never call results."""
import asyncio
import base64
from dataclasses import replace
import json
import time

import httpx
import pytest

from lab_arena import broker as br, contracts, operations, reasoning_capture, training_capture
from lab_arena.training_export import _calls, assemble_example
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, HOST_KEYS, LUNA_RESPONSES, _async_client_factory, luna_price_table, make_broker,
)
from tests.lab_arena.training_export_test import Objects, add_call, fixture
from tests.lab_arena.reasoning_capture_test import _reviewed_registry


BODY = json.dumps({
    "id": "fixture", "status": "completed", "output": [{
        "type": "reasoning", "summary": [],
        "content": [{"type": "reasoning_text", "text": "safe fixture thought"}],
    }], "usage": {"cost": "0.0000123", "output_tokens_details": {"reasoning_tokens": 1}},
}, separators=(",", ":")).encode()


class BrokenStream(httpx.AsyncByteStream):
    def __init__(self, payload, failure):
        self.payload, self.failure = payload, failure
        self.closed = False

    async def __aiter__(self):
        yield self.payload
        if self.failure == "deadline":
            await asyncio.Event().wait()
        elif self.failure is not None:
            raise self.failure("private synthetic transport detail")

    async def aclose(self):
        self.closed = True


def transport_fixture(payload, failure=httpx.ReadTimeout, status=200, *, generation=False, **kwargs):
    stream = BrokenStream(payload, failure)
    sent = []

    def respond(request):
        sent.append(request)
        return httpx.Response(status, headers={"content-type": "application/json",
            **({"X-Generation-Id": "gen-private-timeout-fixture"} if generation else {})}, stream=stream)

    return br.HttpxProviderTransport(client_factory=_async_client_factory(respond), **kwargs), stream, sent


def failed_call(payload, failure=httpx.ReadTimeout, status=200, *, capture=None, generation=False,
                wire_fixture=None, timeout_ms=5000):
    transport, stream, sent = wire_fixture or transport_fixture(payload, failure, status, generation=generation)
    objects = Objects()
    broker, store, _ = make_broker(transport=transport,
        capture_request=capture or training_capture.RequestCapture(objects).prepare)
    broker._price_table = luna_price_table()
    args = dict(operation_id="openrouter.responses", parameters=LUNA_RESPONSES,
                action_sequence=0, timeout_ms=timeout_ms)
    context = replace(CONTEXT, reasoning_capture_policy=reasoning_capture.POLICY,
        reasoning_route_registry=_reviewed_registry(model=LUNA_RESPONSES["model"], api="openrouter.responses"))
    try:
        result = broker.execute(context, **args)
        saved = store.calls[result.call["call_identity"]]
        replay = broker.execute(context, **args)
    finally:
        transport.close()
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert result.call["outcome"] == "uncertain" and "actual_microusd" not in result.call
    assert replay.status == 409 and replay.call["error_code"] == "call_uncertain"
    assert len(sent) == 1
    if wire_fixture is None:
        assert stream.closed
    assert saved["kind"] == "uncertain" and "actual" not in saved
    doc = saved["uncertain_doc"]
    assert len(json.dumps(doc).encode()) <= 4 * 1024 * 1024
    assert doc["reason"] == ("missing_provider_cost" if generation else "transport_failure")
    assert doc["call_succeeded"] is False
    if generation:
        assert doc["transport_failure"] is True
    assert doc["observed_provider_status"] == status
    observations = br.broker_observations(doc)
    assert observations["timing_scope"] == "initial_provider_transport_send"
    assert 0 <= observations["transport_elapsed_ms"] < 5000
    assert len(objects.values) == (0 if capture is not None else 1)
    assert "safe fixture thought" not in repr(result.to_document())
    return result, doc


@pytest.mark.parametrize("size", [None, 65536, 65537])
@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.RemoteProtocolError])
@pytest.mark.parametrize("status", [200, 503])
def test_complete_observed_json_survives_transport_failure_without_becoming_success(size, failure, status):
    payload = BODY if size is None else BODY + b" " * (size - len(BODY))
    result, doc = failed_call(payload, failure, status)
    receipt = doc["upstream_response_receipt"]
    assert receipt["capture_kind"] == "upstream_response" and receipt["call_succeeded"] is False
    assert receipt["status"] == status
    assert receipt["call_identity"] == result.call["call_identity"]
    assert receipt["request_hash"] == result.call["request_hash"]
    assert base64.b64decode(receipt["body_b64"]) == payload
    assert receipt["wire_body_sha256"] == contracts.hash_bytes(payload)
    assert receipt["wire_body_bytes"] == len(payload)
    assert doc["transport_error_class"] == failure.__name__
    assert not br._has_successful_terminal_response(doc)


@pytest.mark.parametrize("payload", [BODY[:40], b'{"output":[],"output":[]}', b'not JSON', b'[]', b'{"output":[],"usage":NaN}',
                                     b'{"output":[],"usage":Infinity}', b'{"output":[],"usage":-Infinity}'])
def test_malformed_partial_and_duplicate_model_json_keep_existing_unavailable_classification(payload):
    _, doc = failed_call(payload)
    receipt = doc["upstream_response_receipt"]
    assert set(receipt) == {"broker_observations"}
    assert "body_b64" not in receipt


def test_observed_body_privacy_and_active_key_guard_precede_persistence():
    payload = json.dumps({"api_key": "private-other-key", "output": [{"type": "reasoning",
        "encrypted_content": "private-opaque-state", "content": [{"type": "reasoning_text", "text": "safe"}]}]}).encode()
    _, doc = failed_call(payload)
    receipt = doc["upstream_response_receipt"]
    assert receipt["privacy"] == {"redacted": True, "redactions": ["credential", "hidden_reasoning"]}
    assert b"private-other-key" not in base64.b64decode(receipt["body_b64"])
    assert b"private-opaque-state" not in base64.b64decode(receipt["body_b64"])
    # JSON escaping defeats a byte-only check; the existing structured guard must run.
    secret = HOST_KEYS["openrouter"]
    escaped = "".join("\\u%04x" % ord(char) for char in secret)
    _, doc = failed_call(('{"output":[{"text":"' + escaped + '"}]}').encode())
    assert "body_b64" not in doc["upstream_response_receipt"]
    assert secret not in json.dumps(doc)


def test_absolute_deadline_retains_observed_body_and_closes_stream():
    transport, stream, sent = transport_fixture(BODY, "deadline")
    started = time.monotonic()
    try:
        with pytest.raises(br.ProviderTransportError) as raised:
            transport.send(method="POST", url="https://openrouter.ai/api/v1/responses",
                           headers={}, body=b"{}", timeout_seconds=0.03)
    finally:
        transport.close()
    assert str(raised.value) == "ReadTimeout" and raised.value.observed_body == BODY
    assert stream.closed and len(sent) == 1 and time.monotonic() - started < 1
    assert "safe fixture thought" not in repr(raised.value)


def test_external_cancellation_still_propagates_and_closes_stream():
    transport, stream, sent = transport_fixture(BODY, asyncio.CancelledError)
    try:
        with pytest.raises(asyncio.CancelledError):
            transport.send(method="POST", url="https://openrouter.ai/api/v1/responses",
                           headers={}, body=b"{}", timeout_seconds=1)
    finally:
        transport.close()
    assert stream.closed and len(sent) == 1


def test_large_observed_body_is_bounded_and_explicitly_unavailable():
    payload = b'{"output":[]}' + b" " * (3 * 1024 * 1024)
    _, doc = failed_call(payload)
    assert set(doc["upstream_response_receipt"]) == {"broker_observations"}
    assert "body_b64" not in doc["upstream_response_receipt"]
    exc = br.ProviderTransportError("ReadTimeout", observed_body=payload * 2)
    assert len(exc.observed_body) <= 3 * 1024 * 1024 + 1
    assert "output" not in repr(exc)


def test_successful_transport_control_and_limit_are_unchanged():
    for limit, expected in [(len(BODY), BODY), (len(BODY) - 1, operations.GENERIC_UNAVAILABLE_BODY)]:
        transport, stream, sent = transport_fixture(BODY, None, max_response_bytes=limit)
        try:
            response = transport.send(method="POST", url="https://openrouter.ai/api/v1/responses",
                                      headers={}, body=b"{}", timeout_seconds=1)
        finally:
            transport.close()
        assert response.body == expected and stream.closed and len(sent) == 1
        assert response.status == (200 if limit == len(BODY) else 502)


def test_native_export_does_not_promote_observed_completed_json_into_training_response():
    _, doc = failed_call(BODY)
    values = fixture()
    add_call(values, status="uncertain", operation="openrouter.responses")
    reserve, _, terminal = values["ledger_rows"]
    receipt = doc["upstream_response_receipt"]
    receipt.update(call_identity=reserve["call_identity"], request_hash=reserve["entry_doc"]["request_hash"])
    terminal["terminal_response"] = {}
    terminal["entry_doc"] = {"call": dict(doc, upstream_response_receipt=receipt)}
    calls, reasons = _calls(values["ledger_rows"], {reserve["run_id"]}, values["objects"], {})
    call = calls[0]
    assert call["upstream_response_receipts"][0]["response"] == json.loads(BODY)
    assert call["upstream_response_receipts"][0]["delivery_to_worker"] == "not_proved"
    assert "response" not in call and "response_entry_id" not in call
    assert call["actual_microusd"] is None
    assert {"full_response_unavailable", "provider_cost_unresolved"} <= set(reasons)
    result = assemble_example(**values)
    assert result["completeness"]["complete"] is False
    assert result["privacy"]["exact_replay"] is False
    assert "full_response_unavailable" in result["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("payload", [b'authorization=private-token incomplete', b'{"api_key":"private-token',
                                     b'{"value":1,"value":2}', b'<html>unfinished'])
def test_tool_transport_capture_also_rejects_malformed_or_non_json_text(payload):
    transport, stream, sent = transport_fixture(payload)
    broker, store, _ = make_broker(transport=transport)
    try:
        result = broker.execute(CONTEXT, operation_id="scrapingdog.scrape",
            parameters={"url": "https://example.com/about"}, action_sequence=0, timeout_ms=5000)
    finally:
        transport.close()
    assert result.status == 502 and len(sent) == 1 and stream.closed
    saved = store.calls[result.call["call_identity"]]
    terminal = saved.get("terminal") or saved.get("uncertain_doc")
    assert not (terminal.get("upstream_response_receipt") or {}).get("body_b64")
    assert "private-token" not in repr(saved)


def test_fixed_scrapingdog_charge_is_unchanged_with_failed_transport_auxiliary():
    transport, stream, sent = transport_fixture(b'{"page":"safe tool evidence"}')
    broker, store, _ = make_broker(transport=transport)
    try:
        result = broker.execute(CONTEXT, operation_id="scrapingdog.scrape",
            parameters={"url": "https://example.com/about"}, action_sequence=0, timeout_ms=5000)
    finally:
        transport.close()
    saved = store.calls[result.call["call_identity"]]
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert result.call["outcome"] == "settled" and saved["actual"] > 0
    terminal = saved["terminal"]
    assert terminal["call_succeeded"] is False
    assert base64.b64decode(terminal["body_b64"]) == operations.GENERIC_UNAVAILABLE_BODY
    assert json.loads(base64.b64decode(terminal["upstream_response_receipt"]["body_b64"])) == {"page": "safe tool evidence"}
    assert terminal["upstream_response_receipt"]["call_succeeded"] is False
    assert len(sent) == 1 and stream.closed


@pytest.mark.parametrize("known_cost", [True, False])
def test_deepline_exact_charge_recovery_does_not_use_auxiliary_as_completed_response(known_cost, monkeypatch):
    from urllib.parse import unquote
    from tests.lab_arena.deepline_native_billing_identity_unit_test import NATIVE_REQUEST_ID, _score_context
    from tests.lab_arena.test_lab_arena_broker import DL_KEY, FakeLedgerStore
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    payload = b'{"status":"completed","results":["safe auxiliary"],"billing":{"credits_charged":999}}'
    stream = BrokenStream(payload, httpx.ReadTimeout)
    sent = []

    def respond(request):
        sent.append(request)
        if request.method == "POST":
            return httpx.Response(200, headers={"x-vercel-id": "iad1::" + NATIVE_REQUEST_ID}, stream=stream)
        if "/executions/by-key/" in str(request.url):
            key = unquote(str(request.url).rsplit("/", 1)[-1])
            return httpx.Response(200, headers={"x-deepline-idempotency-supported": "true"}, json={
                "executionRecovery": {"idempotencyKey": key, "state": "completed"},
                "toolId": "firecrawl_scrape", "requestId": NATIVE_REQUEST_ID,
            })
        return httpx.Response(200, json={"recent": {"request_id": NATIVE_REQUEST_ID, "entries": [{
            "id": "usage-row-1", "request_id": NATIVE_REQUEST_ID, "provider": "firecrawl",
            "operation": "firecrawl_scrape", "credits": 0.02, "delta": -0.02,
            "charge_state": "posted", "charge_finality": "final", "metadata": {},
        }] if known_cost else "invalid"}})

    transport = br.HttpxProviderTransport(client_factory=_async_client_factory(respond))
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    broker, _, _ = make_broker(store=store, transport=transport,
        credential_for=lambda *_: DL_KEY, funding_source_for=lambda _: "miner_key")
    try:
        result = broker.execute(_score_context(), operation_id="scrapingdog.scrape",
            parameters={"url": "https://example.com/about"}, action_sequence=0, timeout_ms=30_000)
    finally:
        transport.close()
    assert [r.method for r in sent] == ["POST", "GET", "GET", "GET"]
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    saved = store.calls[result.call["call_identity"]]
    assert saved["kind"] == ("settlement" if known_cost else "uncertain")
    assert saved.get("actual") == (2_000 if known_cost else None)
    doc = saved["terminal"] if known_cost else saved["uncertain_doc"]
    assert doc["call_succeeded"] is False
    receipt = doc["upstream_response_receipt"]
    assert receipt["call_succeeded"] is False
    assert base64.b64decode(receipt["body_b64"]) == payload
    assert stream.closed


def test_generation_header_timeout_retains_auxiliary_without_cost_or_extra_dispatch():
    result, doc = failed_call(BODY, generation=True)
    assert doc["openrouter_generation_id"] == "gen-private-timeout-fixture"
    assert base64.b64decode(doc["upstream_response_receipt"]["body_b64"]) == BODY
    assert "gen-private-timeout-fixture" not in repr(result.to_document())


@pytest.mark.parametrize("failure,padding", [("eof", 65536), ("stall", 0)])
def test_real_local_tls_failed_stream_reaches_storage_export_and_native_failed_attempt(tmp_path, failure, padding):
    import shutil
    import ssl
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from tests.lab_arena.transport_failure_auxiliary_semantics_test import with_auxiliary
    from tests.lab_arena.training_provider_output_coverage_test import bank
    from tests.lab_arena.training_eligibility_test import assess_ten
    from tests.lab_arena.training_unavailable_upstream_coverage_test import unavailable_upstream
    from lab_arena import training_steps, training_thinking

    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("OpenSSL executable is unavailable")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=localhost", "-addext", "subjectAltName=IP:127.0.0.1",
        "-keyout", str(key), "-out", str(cert)], capture_output=True, timeout=10, check=True)
    payload = BODY + b" " * max(0, padding - len(BODY))
    stop = threading.Event()
    sent = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        closed = False

        def do_POST(self):
            sent.append(self.command)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Generation-Id", "gen-private-timeout-fixture")
            self.send_header("Content-Length", str(len(payload) + 1))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            if failure == "stall":
                stop.wait(2)
            self.close_connection = True
            Handler.closed = True

        def log_message(self, *_):
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    trusted = ssl.create_default_context(cafile=str(cert))

    async def local_only(request):
        request.url = httpx.URL("https://127.0.0.1:%s/provider" % server.server_port)

    transport = br.HttpxProviderTransport(client_factory=lambda: httpx.AsyncClient(
        verify=trusted, trust_env=False, http1=True, http2=False,
        event_hooks={"request": [local_only]}))
    try:
        # The broker still constructs its ordinary provider request. Only the
        # test client's destination changes, to an isolated real TLS socket.
        result, doc = failed_call(payload, generation=True, wire_fixture=(transport, Handler, sent),
                                  timeout_ms=150 if failure == "stall" else 2000)
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert sent == ["POST"] and Handler.closed
    assert doc["transport_error_class"] == ("ReadTimeout" if failure == "stall" else "RemoteProtocolError")
    assert base64.b64decode(doc["upstream_response_receipt"]["body_b64"]) == payload
    example = with_auxiliary(retained_document=doc)
    call = example["provider_calls"][0]
    assert call["upstream_response_receipts"][0]["wire_body_sha256"] == contracts.hash_bytes(payload)
    assert training_thinking.verified_failed_attempt(example, call)
    assert assess_ten(bank(example)) == assess_ten(bank(unavailable_upstream()))
    targets, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(targets["train"]) + len(targets["eval"]) == 2
    assert all(row["source_refs"][0]["call_identity"] != call["call_identity"]
               for row in targets["train"] + targets["eval"])


@pytest.mark.parametrize("generation", [False, True])
def test_valid_json_at_capture_limit_does_not_add_a_gap_when_inline_envelope_cannot_fit(generation):
    # The body passes the3MiB capture cap. Base64 plus terminal metadata exceeds
    # the existing4MiB ledger cap, so the new optional evidence stays absent.
    payload = BODY + b" " * (3 * 1024 * 1024 - len(BODY))
    _, doc = failed_call(payload, generation=generation)
    assert set(doc["upstream_response_receipt"]) == {"broker_observations"}
    assert not doc["upstream_response_receipt"].get("reason")


def test_failed_stream_retention_checks_each_full_terminal_envelope_without_changing_default_fallback():
    receipt = br._capture_upstream_response(br.ProviderResponse(200, {}, BODY),
        operation_id="openrouter.responses", request_hash="hash", call_identity="call", call_succeeded=False)
    original = {"reason": "transport_failure", "call_succeeded": False, "large_accounting": "x" * (4 * 1024 * 1024 - 500)}
    doc = dict(original)
    br._retain_upstream_response_receipt(doc, receipt, unavailable_on_overflow=False)
    assert doc == original
    br._retain_upstream_response_receipt(doc, receipt)
    assert doc["upstream_response_receipt"] == {"status": "unavailable", "reason": "response_size_limit"}


def test_native_failed_stream_envelope_reserves_timing_before_optional_body():
    _, small = failed_call(BODY)
    large = BODY + b" " * (3 * 1024 * 1024 - len(BODY))
    receipt = br._capture_upstream_response(br.ProviderResponse(200, {}, large),
        operation_id="openrouter.responses", request_hash=small["upstream_response_receipt"]["request_hash"],
        call_identity=small["upstream_response_receipt"]["call_identity"], call_succeeded=False)
    before_timing = {key: value for key, value in small.items() if key != "upstream_response_receipt"}
    overhead = len(json.dumps(dict(before_timing, upstream_response_receipt=receipt)).encode()) - len(receipt["body_b64"])
    # Fit exactly without timing; adding timing after the fit check would make
    # the ledger document too large. The body must instead remain optional.
    body_size = ((4 * 1024 * 1024 - overhead) // 4) * 3
    payload = BODY + b" " * (body_size - len(BODY))
    _, doc = failed_call(payload)
    assert set(doc["upstream_response_receipt"]) == {"broker_observations"}
    assert len(json.dumps(doc).encode()) <= 4 * 1024 * 1024
