"""Private upstream evidence cannot change errors, paid calls or cost authority."""
import base64
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, contracts, operations, training_capture
from lab_arena.training_export import _calls, assemble_example
from tests.lab_arena.broker_accounting_outage_test import clock_fixture
from tests.lab_arena.deepline_terminal_error_test import ErrorTransport, PARAMETERS, RetainedErrorStore
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import CONTEXT, CHAT, DL_KEY, FakeTransport, make_broker
from tests.lab_arena.training_export_test import Objects, add_call, fixture


@pytest.fixture(autouse=True)
def bounded_billing(monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)


def failed_call(status, body, *, capture=None):
    objects = Objects()
    transport = ErrorTransport(status, body=body)
    broker, store, _ = make_broker(
        transport=transport, store=RetainedErrorStore(), credential_for=lambda *_: DL_KEY,
        capture_request=capture or training_capture.RequestCapture(objects).prepare,
    )
    context = replace(CONTEXT, deepline_catalog=frozen(row("exa_search", provider="exa")))
    args = dict(operation_id="deepline.execute", parameters=PARAMETERS,
                action_sequence=0, timeout_ms=5000)
    result = broker.execute(context, **args)
    return result, broker, store, transport, objects, context, args


def upstream(saved):
    return saved["uncertain_doc"]["upstream_response_receipt"]


@pytest.mark.parametrize("status,body", [
    (502, b'{"request_id":"terminal-error-request","error":{"detail":"full upstream failure"}}'),
    (500, b"<html>complete upstream error</html>"),
    (429, b'{"error":'), (502, b""),
])
def test_failed_upstream_retained_without_changing_generic_reply_or_uncertainty(status, body):
    result, broker, store, transport, objects, context, args = failed_call(status, body)
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert result.call["outcome"] == "uncertain" and "actual_microusd" not in result.call
    saved = store.calls[result.call["call_identity"]]
    assert saved["uncertain_doc"]["reason"] == "missing_provider_cost"
    receipt = upstream(saved)
    assert receipt["capture_kind"] == "upstream_response"
    assert receipt["call_identity"] == result.call["call_identity"]
    assert receipt["request_hash"] == result.call["request_hash"]
    assert receipt["status"] == status
    assert base64.b64decode(receipt["body_b64"]) == body
    assert receipt["wire_body_sha256"] == receipt["body_sha256"] == contracts.hash_bytes(body)
    assert receipt["wire_body_bytes"] == receipt["body_bytes"] == len(body)
    assert len(objects.values) == 1  # Existing request object only; no response storage I/O.
    replay = broker.execute(context, **args)
    assert replay.status == 409 and replay.call["error_code"] == "call_uncertain"
    assert [r["method"] for r in transport.sent].count("POST") == 1
    assert saved["kind"] == "uncertain"
    assert "full upstream failure" not in repr(result.to_document())


def test_json_credentials_opaque_reasoning_and_plain_malformed_secrets_are_filtered():
    body = json.dumps({"api_key": "private-other-key", "error": {"detail": "keep"},
                       "encrypted_content": "private-opaque-state"}).encode()
    result, _, store, _, _, _, _ = failed_call(502, body)
    receipt = upstream(store.calls[result.call["call_identity"]])
    raw = base64.b64decode(receipt["body_b64"])
    assert b"private-other-key" not in raw and b"private-opaque-state" not in raw
    assert json.loads(raw)["error"]["detail"] == "keep"
    assert receipt["privacy"] == {"redacted": True, "redactions": ["credential", "hidden_reasoning"]}
    result, _, store, _, _, _, _ = failed_call(502, b"authorization=private-credential invalid-json")
    receipt = upstream(store.calls[result.call["call_identity"]])
    assert b"private-credential" not in base64.b64decode(receipt["body_b64"])
    assert receipt["privacy"]["redacted"]


def test_binary_and_oversized_evidence_are_explicit_without_clipping(monkeypatch):
    for body in [b"\xff\x00", b"x" * (3 * 1024 * 1024 + 1)]:
        result, _, store, _, _, _, _ = failed_call(502, body)
        receipt = upstream(store.calls[result.call["call_identity"]])
        assert receipt["status"] == "unavailable" and "body_b64" not in receipt
        assert result.status == 502 and result.call["outcome"] == "uncertain"


def test_capture_outage_cannot_remove_inline_evidence_or_change_accounting():
    def broken(*args, **kwargs):
        raise RuntimeError("private capture failure")
    result, _, store, transport, _, _, _ = failed_call(502, b'{"error":"failed"}', capture=broken)
    saved = store.calls[result.call["call_identity"]]
    assert saved["kind"] == "uncertain"
    assert base64.b64decode(upstream(saved)["body_b64"]) == b'{"error":"failed"}'
    assert result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert [r["method"] for r in transport.sent].count("POST") == 1
    assert "private capture failure" not in repr(saved)


@pytest.mark.parametrize("known_cost", [False, True])
def test_delayed_response_storage_cannot_consume_accounting_deadline(monkeypatch, known_cost):
    clock = clock_fixture(monkeypatch)
    class DelayedResponseStore(Objects):
        writes = 0
        def put(self, ref, value):
            self.writes += 1
            if self.writes > 1:
                clock.now += 240  # A delayed, nonthrowing storage response.
            super().put(ref, value)
    objects = DelayedResponseStore()
    capture = training_capture.RequestCapture(objects).prepare
    if known_cost:
        broker, store, transport = make_broker(transport=FakeTransport([(500, {
            "usage": {"cost": "0.0000123"}, "error": {"message": "upstream failed"}})]), capture_request=capture)
        result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
                                action_sequence=0, timeout_ms=5000)
        saved = store.calls[result.call["call_identity"]]
        assert saved["kind"] == "settlement" and saved["actual"] == 13
    else:
        result, _, store, transport, _, _, _ = failed_call(502, b'{"error":"failed"}', capture=capture)
        assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"
    assert objects.writes == 1 and clock.now == 0
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY


def test_known_cost_error_keeps_exact_settlement_replay_and_private_receipt():
    body = {"usage": {"cost": "0.0000123"}, "error": {"message": "upstream failed"}}
    broker, store, transport = make_broker(transport=FakeTransport([(500, body)]))
    args = dict(operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    result = broker.execute(CONTEXT, **args)
    saved = store.calls[result.call["call_identity"]]
    assert result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert saved["kind"] == "settlement" and saved["actual"] == 13
    terminal = saved["terminal"]
    assert base64.b64decode(terminal["body_b64"]) == operations.GENERIC_UNAVAILABLE_BODY
    assert json.loads(base64.b64decode(terminal["upstream_response_receipt"]["body_b64"])) == body
    replay = broker.execute(CONTEXT, **args)
    assert replay.status == result.status and replay.body == result.body
    assert replay.call["actual_microusd"] == 13 and len(transport.sent) == 1
    for reply in (result, replay):
        assert "upstream_response_receipt" not in repr(reply.to_document())
        assert "upstream failed" not in repr(reply.to_document())


@pytest.mark.parametrize("wire_status,body", [
    (402, {"error": {"code": 402, "message": "in flight", "metadata": {"limit_source": "openrouter_in_flight_budget"}}}),
    (200, {"error": {"code": 429, "message": "upstream limited"}, "usage": {"cost": "0"}}),
])
def test_openrouter_capture_preserves_wire_status_before_effective_status(wire_status, body):
    broker, store, _ = make_broker(transport=FakeTransport([(wire_status, body)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    saved = store.calls[result.call["call_identity"]]
    receipt = saved.get("terminal", {}).get("upstream_response_receipt") or upstream(saved)
    assert receipt["status"] == wire_status
    assert json.loads(base64.b64decode(receipt["body_b64"])) == body
    assert result.status == 502  # Effective throttles still use the existing bridge error.


def test_success_and_full_failure_bytes_are_unchanged_and_secret_guard_skips_capture():
    objects = Objects()
    body = {"model": "openai/gpt-4o-mini", "choices": [], "usage": {"cost": "0.0000123"}}
    broker, store, transport = make_broker(transport=FakeTransport([(200, body)]),
        capture_request=training_capture.RequestCapture(objects).prepare)
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.status == 200 and json.loads(result.body) == body
    assert "upstream_response_receipt" not in store.calls[result.call["call_identity"]]["terminal"]
    assert len(objects.values) == len(transport.sent) == 1
    long_body = b'{"error":  "' + b"x" * 20000 + b'" }\n'
    result, _, store, _, _, _, _ = failed_call(502, long_body)
    assert base64.b64decode(upstream(store.calls[result.call["call_identity"]])["body_b64"]) == long_body
    assert br._capture_upstream_response(br.ProviderResponse(502, {}, operations.GENERIC_UNAVAILABLE_BODY, "credential_echo"),
        operation_id="deepline.execute", request_hash="hash", call_identity="call") is None


def export_fixture(*, redacted=False):
    values = fixture()
    add_call(values, status="uncertain", provider="deepline", operation="deepline.execute")
    reserve, _, terminal = values["ledger_rows"]
    error = {"error": {"detail": "upstream diagnostic"},
             **({"api_key": "private-key", "encrypted_content": "private-opaque-state"} if redacted else {})}
    receipt = br._capture_upstream_response(br.ProviderResponse(502, {}, json.dumps(error).encode()),
        operation_id=reserve["operation_id"], call_identity=reserve["call_identity"], request_hash=reserve["entry_doc"]["request_hash"])
    terminal["entry_doc"] = {"call": {"upstream_response_receipt": receipt}}
    terminal["terminal_response"] = {"status": 502, "body_b64": base64.b64encode(operations.GENERIC_UNAVAILABLE_BODY).decode()}
    return values, reserve, terminal, receipt


@pytest.mark.parametrize("redacted", [False, True])
def test_export_separates_upstream_error_from_bridge_response_and_unknown_price(redacted):
    values, reserve, terminal, _ = export_fixture(redacted=redacted)
    calls, reasons = _calls(values["ledger_rows"], {reserve["run_id"]}, values["objects"], {})
    call = calls[0]
    assert call["response"] == json.loads(operations.GENERIC_UNAVAILABLE_BODY)
    receipt = call["upstream_response_receipts"][0]
    assert receipt["response"]["error"]["detail"] == "upstream diagnostic"
    assert receipt["delivery_to_worker"] == "not_proved"
    assert call["actual_microusd"] is None and "provider_cost_unresolved" in reasons
    assert ("upstream_response_privacy_nonexact" in reasons) is redacted
    assert "private-key" not in json.dumps(call)
    if redacted:
        exported = assemble_example(**values)
        assert exported["privacy"]["privacy_filtered"]
        assert exported["privacy"]["hidden_reasoning_removed"]
        assert "private-opaque-state" not in json.dumps(exported)
    terminal["terminal_response"] = {}
    calls, reasons = _calls(values["ledger_rows"], {reserve["run_id"]}, values["objects"], {})
    assert "response" not in calls[0] and "full_response_unavailable" in reasons
    assert calls[0]["upstream_response_receipts"]


@pytest.mark.parametrize("mutation", ["wrong_call", "wrong_request", "body_corrupted", "unavailable"])
def test_export_rejects_unbound_corrupt_or_unavailable_upstream_receipt(mutation):
    values, reserve, terminal, receipt = export_fixture()
    if mutation == "wrong_call": receipt["call_identity"] = "wrong"
    elif mutation == "wrong_request": receipt["request_hash"] = "wrong"
    elif mutation == "body_corrupted": receipt["body_b64"] = base64.b64encode(b"changed").decode()
    else: terminal["entry_doc"]["call"]["upstream_response_receipt"] = {"status": "unavailable", "reason": "response_size_limit"}
    calls, reasons = _calls(values["ledger_rows"], {reserve["run_id"]}, values["objects"], {})
    assert "upstream_response_capture_unavailable" in reasons
    assert "upstream_response_receipts" not in calls[0]
    if mutation == "unavailable":
        assert "upstream_response_size_limit" in reasons


def test_transport_failure_does_not_claim_an_upstream_response():
    broker, store, _ = make_broker(transport=FakeTransport(fail=True))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.status == 502
    assert "upstream_response_receipt" not in store.calls[result.call["call_identity"]]["uncertain_doc"]


def test_malformed_model_response_keeps_private_text_without_final_price():
    class Malformed(FakeTransport):
        def send(self, **kwargs):
            self.sent.append(kwargs)
            return br.ProviderResponse(200, {"content-type": "application/json"}, b'{"choices":')
    broker, store, transport = make_broker(transport=Malformed())
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    saved = store.calls[result.call["call_identity"]]
    assert result.status == 502 and saved["kind"] == "uncertain"
    assert "known_actual_microusd" not in saved["uncertain_doc"]
    assert base64.b64decode(upstream(saved)["body_b64"]) == b'{"choices":'
    assert len(transport.sent) == 1


def test_existing_large_agent_receipt_leaves_explicit_remaining_capacity_gap():
    body = json.dumps({"request_id": "terminal-error-request", "error": "x" * 1700000}).encode()
    result, _, store, _, _, _, _ = failed_call(422, body)
    assert result.status == 422 and result.body == body
    saved = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert base64.b64decode(saved["deepline_terminal_response"]["body_b64"]) == body
    assert saved["upstream_response_receipt"] == {"status": "unavailable", "reason": "response_size_limit"}
    assert len(json.dumps(saved).encode()) <= 4 * 1024 * 1024


def test_nul_stays_base64_safe_and_headers_are_privacy_filtered():
    body = b'{"error":"before\\u0000after"}'
    receipt = br._capture_upstream_response(br.ProviderResponse(502,
        {"content-type": "text/plain; api_key=private-header-key", "set-cookie": "private-cookie"}, body),
        operation_id="deepline.execute", request_hash="hash", call_identity="call")
    assert base64.b64decode(receipt["body_b64"]) == body
    assert "\\u0000" not in json.dumps(receipt)
    assert "private-header-key" not in json.dumps(receipt)
    assert "private-cookie" not in json.dumps(receipt)
    assert receipt["privacy"]["redacted"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_generic_response_metadata_never_contains_private_upstream_receipt(corrupt):
    values, reserve, terminal, receipt = export_fixture()
    terminal["entry_doc"] = {}
    terminal["terminal_response"]["upstream_response_receipt"] = receipt
    if corrupt:
        receipt["body_b64"] = base64.b64encode(b"private rejected upstream bytes").decode()
    calls, reasons = _calls(values["ledger_rows"], {reserve["run_id"]}, values["objects"], {})
    metadata = calls[0]["response_receipts"][0]["metadata"]
    assert "upstream_response_receipt" not in metadata
    assert receipt["body_b64"] not in json.dumps(metadata)
    assert calls[0]["response"] == json.loads(operations.GENERIC_UNAVAILABLE_BODY)
    if corrupt:
        assert "upstream_response_capture_unavailable" in reasons
        assert "upstream_response_receipts" not in calls[0]
        assert receipt["body_b64"] not in json.dumps(calls)
    else:
        assert calls[0]["upstream_response_receipts"][0]["response"]["error"]["detail"] == "upstream diagnostic"


def test_full_accounting_document_is_preserved_when_even_gap_marker_cannot_fit():
    limit = 4 * 1024 * 1024
    document = {"padding": "x" * (limit - len(json.dumps({"padding": ""}).encode()))}
    original = dict(document)
    assert len(json.dumps(document).encode()) == limit
    br._retain_upstream_response_receipt(document, {"status": 502, "body_b64": "large"})
    assert document == original and "upstream_response_receipt" not in document
    assert len(json.dumps(document).encode()) == limit


def test_receipt_capacity_includes_existing_deepline_metadata_and_keeps_accounting():
    limit = 4 * 1024 * 1024
    document = {"status": 502, "call_succeeded": False, "known_actual_microusd": 13,
                "deepline_async_job_ids": ["11111111-2222-4333-8444-555555555555"] * 32,
                "padding": ""}
    # The accounting fields, including all Deepline IDs, leave room only for
    # the explicit small gap marker, never the full optional receipt.
    document["padding"] = "x" * (limit - len(json.dumps(document).encode()) - 150)
    original = dict(document)
    br._retain_upstream_response_receipt(document, {"status": 502, "body_b64": "x" * 200})
    assert {key: value for key, value in document.items() if key != "upstream_response_receipt"} == original
    assert document["upstream_response_receipt"] == {"status": "unavailable", "reason": "response_size_limit"}
    assert len(json.dumps(document).encode()) <= limit
