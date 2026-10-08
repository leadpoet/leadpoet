"""Terminal request errors must not become outages while their bill is pending."""

from dataclasses import replace
import json

import pytest

from lab_arena import broker as br
from lab_arena.service import _provider_telemetry_fields
from tests.lab_arena.deepline_worker_same_action_recovery_test import Api, worker
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import _candidate
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, DL_KEY, ZeroReservationLedgerStore, deepline_generic_http_denial, make_broker,
)


REQUEST_ID = "terminal-error-request"
PARAMETERS = {"tool": "exa_search", "payload": {"query": "example"}}
UNREADABLE_ERROR_BODIES = [
    pytest.param(b"not JSON", id="plain"),
    pytest.param(b"<html>Not Found</html>", id="html"),
    pytest.param(b"", id="empty"),
    pytest.param(b'"scalar"', id="scalar"),
    pytest.param(b"17", id="numeric"),
    pytest.param(b'{"error":', id="malformed-json"),
    pytest.param(b"\xff", id="invalid-utf8"),
]


def billing(*, credits="0", final=True):
    entry = {
        "id": "terminal-error-charge", "request_id": REQUEST_ID,
        "provider": "exa", "operation": "exa_search", "status": "error",
        "charge_state": "posted", "charge_finality": "final" if final else "pending",
        "credits": credits, "delta": "-" + credits,
    }
    return {"recent": {"request_id": REQUEST_ID, "entries": [entry]}}


class ErrorTransport:
    """Return exact supplied receipts; never add billing/finality to a fixture."""

    def __init__(self, status=404, *, bill=None, body=None):
        self.status, self.bill = status, bill
        self.body = body if body is not None else json.dumps({
            "request_id": REQUEST_ID, "error": {"code": "NOT_FOUND"},
        }).encode()
        self.sent = []

    def send(self, **request):
        self.sent.append(request)
        if request["method"] == "POST":
            self.key = request["headers"].get("idempotency-key")
            return br.ProviderResponse(self.status, {"content-type": "application/json"}, self.body)
        assert request["method"] == "GET"
        if "/executions/by-key/" in request["url"]:
            document = {
                "requestId": REQUEST_ID, "toolId": "exa_search",
                "executionRecovery": {"idempotencyKey": self.key, "state": "failed"},
            }
            return br.ProviderResponse(200, {"x-deepline-idempotency-supported": "true"},
                                       json.dumps(document).encode())
        assert request["url"] == br.DEEPLINE_EXACT_BILLING_URL + REQUEST_ID
        document = self.bill if self.bill is not None else {
            "recent": {"request_id": REQUEST_ID, "entries": []},
        }
        return br.ProviderResponse(200, {}, json.dumps(document).encode())



class RetainedErrorStore(ZeroReservationLedgerStore):
    """Expose the same append-only uncertainty history as the real store."""

    def list_ledger(self, *, call_identity=None, **kwargs):
        rows = super().list_ledger(call_identity=call_identity, **kwargs)
        call = self.calls.get(call_identity, {})
        if "uncertain_doc" not in call:
            return rows
        reservation = rows[0]
        history = [reservation, {**reservation, "entry_id": 2, "entry_kind": "dispatch"},
                   {**reservation, "entry_id": 3, "entry_kind": "uncertain",
                    "entry_doc": {"reason": "worker_reported", "call": call["uncertain_doc"]}}]
        if call["kind"] == "settlement":
            history.append({**rows[-1], "entry_id": 4})
        return history

    def _view(self, call):
        state = super()._view(call)
        if call["kind"] == "settlement" and "uncertain_doc" in call:
            state["deepline_response_missing"] = True
        return state


def execute(transport, *, kind="execute", store=None, funding_source=None):
    funding_options = (
        {"provider_funding_source_for": lambda _context, _provider: funding_source}
        if funding_source is not None else {}
    )
    broker, store, _ = make_broker(
        transport=transport, store=store or RetainedErrorStore(),
        credential_for=lambda *_: DL_KEY,
        **funding_options,
    )
    arguments = dict(operation_id="deepline.execute", parameters=PARAMETERS,
                     action_sequence=0, timeout_ms=30_000)
    context = replace(CONTEXT, kind=kind,
                      deepline_catalog=frozen(row("exa_search", provider="exa")))
    return broker.execute(context, **arguments), broker, store, context, arguments


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("status", [400, 404, 422])
def test_pending_error_returns_original_status_without_polling_or_worker_retry(
    monkeypatch, tmp_path, kind, status,
):
    monkeypatch.setattr(br.time, "sleep", lambda _: pytest.fail("terminal error must not poll"))
    transport = ErrorTransport(status)
    result, broker, store, context, arguments = execute(transport, kind=kind)

    assert result.status == status and result.body == transport.body
    assert result.call["outcome"] == "uncertain"
    assert result.call["provider_status"] == status
    assert "actual_microusd" not in result.call
    assert "error_code" not in result.call
    assert store.log == ["reserve", "dispatch", "uncertain"]
    saved = store.calls[result.call["call_identity"]]
    assert saved["uncertain_doc"]["reason"] == "missing_provider_cost"
    assert saved["uncertain_doc"]["call_succeeded"] is False
    assert saved["uncertain_doc"]["deepline_job_id"] == REQUEST_ID
    assert [r["method"] for r in transport.sent] == ["POST", "GET"]
    assert 0 < transport.sent[1]["timeout_seconds"] <= 5
    telemetry = _provider_telemetry_fields(result.call, result.status, 1)
    assert telemetry["http_status"] == telemetry["provider_status"] == status
    assert telemetry["outcome"] == "uncertain"  # Unknown billing remains visible.

    api = Api(result.to_document())
    host = worker(tmp_path, api)
    error, returned = host._dispatch_once("deepline.execute", PARAMETERS, 30_000)
    assert error is None and returned["status"] == status
    assert len(api.frames) == 1

    # Even an explicit replay cannot create another paid call or free settlement.
    replay = broker.execute(context, **arguments)
    assert replay.status == status and replay.body == result.body
    assert replay.call["idempotent"] is True
    assert "actual_microusd" not in replay.call
    assert len(transport.sent) == 2
    assert [r["method"] for r in transport.sent].count("POST") == 1
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"


@pytest.mark.parametrize("status", [400, 404, 422])
def test_list_client_rejection_preserves_body_and_safe_replay(monkeypatch, status):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    transport = ErrorTransport(status, body=b'[{"error":"Invalid tool"}]')
    result, broker, store, context, arguments = execute(transport)
    assert result.status == status and result.body == transport.body
    assert result.call["outcome"] == "uncertain" and "actual_microusd" not in result.call
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == status and replay.body == result.body
    assert replay.call["idempotent"] is True and replay.call["outcome"] == "uncertain"
    assert "actual_microusd" not in replay.call and len(transport.sent) == before
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"


@pytest.mark.parametrize("bill", [billing(final=False), {"recent": "invalid"}])
def test_unproven_receipts_never_become_zero_cost(bill):
    result, _, store, _, _ = execute(ErrorTransport(422, bill=bill))
    assert result.status == 422 and result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"


@pytest.mark.parametrize("credits,amount", [("0", 0), ("0.5", 50_000)])
def test_immediate_final_error_receipt_settles_exact_charge(credits, amount):
    transport = ErrorTransport(404, bill=billing(credits=credits))
    result, _, store, _, _ = execute(transport)
    assert result.status == 404 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == amount
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is False
    assert [r["method"] for r in transport.sent] == ["POST", "GET"]


def test_missing_native_id_gets_one_key_lookup_and_one_exact_bill(monkeypatch):
    monkeypatch.setattr(br.time, "sleep", lambda _: pytest.fail("terminal error must not poll"))
    transport = ErrorTransport(body=b'{"error":{"code":"NOT_FOUND"}}')
    result, _, store, _, _ = execute(transport)
    assert result.status == 404
    assert [r["method"] for r in transport.sent] == ["POST", "GET", "GET"]
    assert all(0 < r["timeout_seconds"] <= 5 for r in transport.sent[1:])
    assert store.calls[result.call["call_identity"]]["uncertain_doc"]["deepline_job_id"] == REQUEST_ID


@pytest.mark.parametrize("state", ["stale", "unavailable"])
def test_error_response_requires_durable_uncertainty(state):
    class UnavailableStore(ZeroReservationLedgerStore):
        def mark_uncertain(self, **kwargs):
            return {"status": state}

    result, _, _, _, _ = execute(ErrorTransport(), store=UnavailableStore())
    assert result.status == 502
    assert result.call["error_code"] == "provider_unavailable"


@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
def test_unproven_402_returns_after_one_short_read_without_paid_replay(
    monkeypatch, funding_source,
):
    monkeypatch.setattr(br.time, "sleep", lambda _: pytest.fail("402 must not poll"))
    transport = ErrorTransport(402)
    result, broker, store, context, arguments = execute(
        transport, funding_source=funding_source,
    )
    assert result.status == (402 if funding_source == "miner_key" else 502)
    assert result.call["error_code"] == (
        "miner_credentials_unavailable" if funding_source == "miner_key"
        else "provider_unavailable"
    )
    assert result.call["provider_status"] == 402
    assert result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    assert store.log == ["reserve", "dispatch", "uncertain"]
    saved = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert saved["call_succeeded"] is False
    assert saved["deepline_job_id"] == REQUEST_ID
    assert saved["deepline_execution_key"]
    assert saved["credential_fingerprint"]
    assert [request["method"] for request in transport.sent] == ["POST", "GET"]
    assert 0 < transport.sent[1]["timeout_seconds"] <= 5

    replay = broker.execute(context, **arguments)
    assert replay.status == 409
    assert replay.call["error_code"] == "call_uncertain"
    assert replay.call["outcome"] == "uncertain"
    assert "actual_microusd" not in replay.call
    assert [request["method"] for request in transport.sent].count("POST") == 1


def test_402_short_read_settles_only_exact_final_charge(monkeypatch):
    monkeypatch.setattr(br.time, "sleep", lambda _: pytest.fail("402 must not poll"))
    transport = ErrorTransport(402, bill=billing(credits="0.5"))
    result, _, store, _, _ = execute(transport)
    assert result.status == 502
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 50_000
    assert store.log == ["reserve", "dispatch", "settle"]
    assert [request["method"] for request in transport.sent] == ["POST", "GET"]
    assert 0 < transport.sent[1]["timeout_seconds"] <= 5


def test_complete_non_json_402_still_uses_short_read(monkeypatch):
    monkeypatch.setattr(br.time, "sleep", lambda _: pytest.fail("402 must not poll"))
    transport = ErrorTransport(402, body=b"<html>Payment Required</html>")
    result, _, store, _, _ = execute(transport, funding_source="miner_key")
    assert result.status == 402
    assert result.call["error_code"] == "miner_credentials_unavailable"
    assert result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert [request["method"] for request in transport.sent] == ["POST", "GET", "GET"]
    assert all(0 < request["timeout_seconds"] <= 5 for request in transport.sent[1:])


def test_synthetic_402_keeps_normal_billing_poll(monkeypatch):
    sleeps = []
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(br.time, "sleep", sleeps.append)

    class SyntheticTransport(ErrorTransport):
        def send(self, **request):
            response = super().send(**request)
            if request["method"] == "POST":
                return replace(response, internal_provenance="response_too_large")
            return response

    transport = SyntheticTransport(402)
    result, _, store, _, _ = execute(transport)
    assert result.status == 502
    assert result.call["outcome"] == "uncertain"
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert [request["method"] for request in transport.sent] == ["POST", "GET", "GET"]
    assert len(sleeps) == 1


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502, 504])
def test_account_and_infrastructure_failures_keep_normal_polling(monkeypatch, status):
    sleeps = []
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(br.time, "sleep", sleeps.append)
    transport = ErrorTransport(status)
    result, _, _, _, _ = execute(transport)
    assert result.status == 502 and result.call["error_code"] == "provider_unavailable"
    assert [r["method"] for r in transport.sent] == ["POST", "GET", "GET"]
    assert len(sleeps) == 1


def test_credential_echo_error_is_not_exposed(monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    result, _, _, _, _ = execute(ErrorTransport(body=json.dumps({"error": DL_KEY}).encode()))
    assert result.status == 502 and result.call["error_code"] == "provider_unavailable"
    assert DL_KEY not in json.dumps(result.to_document())


@pytest.mark.parametrize("status", [400, 404, 422])
@pytest.mark.parametrize("body", UNREADABLE_ERROR_BODIES)
def test_unreadable_client_rejection_keeps_status_and_safe_replay(monkeypatch, status, body):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    transport = ErrorTransport(status, body=body)
    result, broker, store, context, arguments = execute(transport)
    assert result.status == status
    assert isinstance(json.loads(result.body).get("error"), dict)
    assert result.body != body
    assert result.headers["content-type"] == "application/json"
    assert result.call["outcome"] == "uncertain"
    assert result.call["provider_status"] == status
    assert "actual_microusd" not in result.call
    saved = store.calls[result.call["call_identity"]]["uncertain_doc"]
    terminal = saved["deepline_terminal_response"]
    assert terminal["status"] == status and terminal["call_succeeded"] is False
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == status and replay.body == result.body
    assert replay.call["idempotent"] is True and replay.call["outcome"] == "uncertain"
    assert "actual_microusd" not in replay.call
    assert len(transport.sent) == before
    assert [r["method"] for r in transport.sent].count("POST") == 1


@pytest.mark.parametrize("status", [400, 404, 422])
@pytest.mark.parametrize("credits,amount", [("0", 0), ("0.5", 50_000)])
def test_unreadable_rejection_with_exact_bill_keeps_status_and_real_charge(
    monkeypatch, status, credits, amount,
):
    monkeypatch.setattr(br.time, "sleep", lambda _: pytest.fail("final exact receipt must not poll"))
    transport = ErrorTransport(status, body=b"<html>Invalid request</html>", bill=billing(credits=credits))
    result, broker, store, context, arguments = execute(transport)
    assert result.status == status and isinstance(json.loads(result.body).get("error"), dict)
    assert result.call["outcome"] == "settled" and result.call["actual_microusd"] == amount
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert terminal["status"] == status and terminal["call_succeeded"] is False
    assert [r["method"] for r in transport.sent] == ["POST", "GET", "GET"]
    replay = broker.execute(context, **arguments)
    assert replay.status == status and replay.body == result.body
    assert replay.call["actual_microusd"] == amount and replay.call["idempotent"] is True
    assert len(transport.sent) == 3


@pytest.mark.parametrize("credits,amount", [("0", 0), ("0.5", 50_000)])
def test_unreadable_rejection_still_polls_until_exact_final_charge(monkeypatch, credits, amount):
    sleeps = []
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(br.time, "sleep", sleeps.append)

    class DelayedBillTransport(ErrorTransport):
        bill_reads = 0

        def send(self, **request):
            if "/billing/usage?request_id=" in request["url"]:
                self.bill_reads += 1
                self.bill = billing(credits=credits, final=self.bill_reads == 3)
            return super().send(**request)

    transport = DelayedBillTransport(422, body=b"Invalid request")
    result, broker, store, context, arguments = execute(transport)
    assert result.status == 422 and isinstance(json.loads(result.body).get("error"), dict)
    assert result.call["outcome"] == "settled" and result.call["actual_microusd"] == amount
    assert store.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is False
    assert transport.bill_reads == 3 and len(sleeps) == 2
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == 422 and replay.body == result.body
    assert replay.call["actual_microusd"] == amount and replay.call["idempotent"] is True
    assert len(transport.sent) == before


class SettlementBeforeUncertainReturnStore(RetainedErrorStore):
    settled_amount = 0

    def mark_uncertain(self, **kwargs):
        super().mark_uncertain(**kwargs)
        call = self.calls[kwargs["call_identity"]]
        call.update(
            kind="settlement", actual=self.settled_amount,
            terminal=br._terminal_response_document(
                502, {"content-type": "application/json"},
                br.operations.GENERIC_UNAVAILABLE_BODY, call_succeeded=False),
        )
        return self._view(call)


@pytest.mark.parametrize("status", [400, 404, 422])
@pytest.mark.parametrize("amount", [0, 50_000])
def test_raw_rejection_settled_before_uncertain_return_keeps_body_and_charge(monkeypatch, status, amount):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    store = SettlementBeforeUncertainReturnStore()
    store.settled_amount = amount
    transport = ErrorTransport(status, body=b"Invalid request")
    result, broker, _, context, arguments = execute(transport, store=store)
    assert result.status == status and isinstance(json.loads(result.body).get("error"), dict)
    assert result.call["outcome"] == "settled" and result.call["actual_microusd"] == amount
    before = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == status and replay.body == result.body
    assert replay.call["outcome"] == "settled" and replay.call["actual_microusd"] == amount
    assert replay.call["idempotent"] is True and len(transport.sent) == before
    assert sum(r["method"] == "POST" for r in transport.sent) == 1


@pytest.mark.parametrize("amount", [None, -1, True, "0", 0.5])
def test_rejection_settled_before_uncertain_return_rejects_invalid_cost(monkeypatch, amount):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    store = SettlementBeforeUncertainReturnStore()
    store.settled_amount = amount
    result, _, _, _, _ = execute(ErrorTransport(404), store=store)
    assert result.status == 503 and result.call["error_code"] == "broker_unavailable"
    assert "actual_microusd" not in result.call


@pytest.mark.parametrize("status", [401, 402, 403, 429, 500, 502, 504])
def test_unreadable_non_request_failure_stays_unavailable(monkeypatch, status):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    result, _, _, _, _ = execute(ErrorTransport(status, body=b"<html>provider failure</html>"))
    assert result.status == 502 and result.call["error_code"] == "provider_unavailable"
    assert result.call["outcome"] == "uncertain" and "actual_microusd" not in result.call


@pytest.mark.parametrize("status", [402, 422])
def test_late_final_charge_uses_retained_identity_without_another_paid_request(status):
    class ReconciledStore(ZeroReservationLedgerStore):
        def reconcile_deepline_cost(self, **kwargs):
            self.reconciled = kwargs
            return {"status": "settled", "actual_microusd": kwargs["actual_microusd"]}

    transport = ErrorTransport(status)
    result, broker, store, _, _ = execute(transport, store=ReconciledStore())
    assert result.status == (502 if status == 402 else 422)
    identity = result.call["call_identity"]
    saved = store.calls[identity]["uncertain_doc"]
    transport.bill = billing(credits="0.5")
    candidate = _candidate(
        call_identity=identity, request_id=saved["deepline_job_id"],
        execution_key=saved["deepline_execution_key"],
        credential_fingerprint=saved["credential_fingerprint"],
        operation=saved["deepline_operation"], billing_provider="exa",
    )
    assert broker.reconcile_deepline_cost(candidate)["actual_microusd"] == 50_000
    assert store.reconciled["call_identity"] == identity
    assert store.reconciled["recovered_request_id"] == REQUEST_ID
    assert store.reconciled["credential_fingerprint"] == br._credential_fingerprint(DL_KEY)
    assert [r["method"] for r in transport.sent] == ["POST", "GET", "GET"]


def test_key_lookup_cannot_start_a_bill_read_after_the_short_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(br.time, "monotonic", lambda: now[0])

    class SlowKeyTransport(ErrorTransport):
        def send(self, **request):
            response = super().send(**request)
            if "/executions/by-key/" in request["url"]:
                now[0] += 5
            return response

    transport = SlowKeyTransport(body=b'{"error":{"code":"NOT_FOUND"}}')
    result, _, _, _, _ = execute(transport)
    assert result.status == 404 and result.call["outcome"] == "uncertain"
    assert [r["method"] for r in transport.sent] == ["POST", "GET"]


@pytest.mark.parametrize("field,value", [("run_id", "foreign"), ("provider", "openrouter"),
                                         ("operation_id", "exa.contents"), ("amount_microusd", 1)])
def test_retained_error_requires_bound_ledger_history(field, value):
    class CorruptStore(RetainedErrorStore):
        def list_ledger(self, **kwargs):
            rows = super().list_ledger(**kwargs)
            rows[0][field] = value
            return rows

    transport = ErrorTransport()
    result, broker, _, context, arguments = execute(transport, store=CorruptStore())
    assert result.status == 404
    replay = broker.execute(context, **arguments)
    assert replay.status == 503 and replay.call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == 2


@pytest.mark.parametrize("change", [{"status": 200}, {"call_succeeded": True}, {"body_b64": "invalid!"}])
def test_malformed_retained_error_fails_closed(change):
    transport = ErrorTransport()
    result, broker, store, context, arguments = execute(transport)
    store.calls[result.call["call_identity"]]["uncertain_doc"]["deepline_terminal_response"].update(change)
    replay = broker.execute(context, **arguments)
    assert replay.status == 503 and replay.call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == 2


@pytest.mark.parametrize("amount", [0, 50_000])
def test_pending_refusal_replays_only_the_generic_sanitized_body(amount):
    transport = ErrorTransport(403, body=json.dumps(deepline_generic_http_denial(REQUEST_ID)).encode())
    broker, store, _ = make_broker(transport=transport, store=RetainedErrorStore(),
                                  credential_for=lambda *_: DL_KEY)
    arguments = dict(operation_id="deepline.execute", action_sequence=0, timeout_ms=5000,
                     parameters={"tool": "generic_http_request", "payload": {"url": "https://example.com/"}})
    result = broker.execute(CONTEXT, **arguments)
    replay = broker.execute(CONTEXT, **arguments)
    assert result.status == replay.status == 403
    assert result.body == replay.body == b'{"error":{"code":"provider_request_refused"}}'
    assert replay.call["outcome"] == "uncertain" and replay.call["idempotent"] is True
    assert replay.call["error_code"] == "provider_request_refused"
    assert "actual_microusd" not in replay.call
    assert "private" not in repr(store.calls)
    assert len(transport.sent) == 2

    call = store.calls[result.call["call_identity"]]
    call.update(kind="settlement", actual=amount, terminal=br._terminal_response_document(
        502, {"content-type": "application/json"}, b'{"error":{"code":"provider_unavailable"}}',
        call_succeeded=False))
    settled = broker.execute(CONTEXT, **arguments)
    assert settled.status == 403 and settled.body == result.body
    assert settled.call["outcome"] == "settled" and settled.call["actual_microusd"] == amount
    assert settled.call["error_code"] == "provider_request_refused" and settled.call["idempotent"] is True
    assert len(transport.sent) == 2


@pytest.mark.parametrize("status", [400, 404, 422])
def test_unkeyed_batch_error_replays_without_execution_lookup(status):
    transport = ErrorTransport(status)
    broker, store, _ = make_broker(transport=transport, store=RetainedErrorStore(),
                                  credential_for=lambda *_: DL_KEY)
    context = replace(CONTEXT, deepline_catalog=frozen(row("firecrawl_batch_scrape", provider="firecrawl")))
    arguments = dict(operation_id="deepline.execute", action_sequence=0, timeout_ms=5000,
                     parameters={"tool": "firecrawl_batch_scrape", "payload": {"query": "example"}})
    result = broker.execute(context, **arguments)
    assert result.status == status and transport.key is None
    pending = broker.execute(context, **arguments)
    assert pending.status == status and pending.body == result.body
    call = store.calls[result.call["call_identity"]]
    call.update(kind="settlement", actual=50_000, terminal=br._terminal_response_document(
        502, {}, b'{"error":{"code":"provider_unavailable"}}', call_succeeded=False))
    settled = broker.execute(context, **arguments)
    assert settled.status == status and settled.body == result.body
    assert settled.call["actual_microusd"] == 50_000 and settled.call["idempotent"] is True
    assert [request["method"] for request in transport.sent] == ["POST", "GET"]


def test_oversized_error_is_not_retained_for_replay(monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    body = json.dumps({"request_id": REQUEST_ID, "error": "x" * 1_048_576}).encode()
    result, _, store, _, _ = execute(ErrorTransport(body=body))
    assert result.status == 502 and result.call["error_code"] == "provider_unavailable"
    saved = store.calls[result.call["call_identity"]]["uncertain_doc"]
    assert "deepline_terminal_response" not in saved


@pytest.mark.parametrize("successful", [False, True])
@pytest.mark.parametrize("refresh_patch", [None, {"call_identity": "foreign"},
                                         {"amount_microusd": 1}, {"status": "uncertain"}])
def test_settlement_racing_replay_refreshes_once_and_still_requires_bound_state(
    monkeypatch, refresh_patch, successful,
):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    saved_body = b'{"results":[{"url":"https://example.com/"}]}'

    class ConcurrentSettlementStore(RetainedErrorStore):
        settle_during_read = False
        refreshed = False
        reservation_reads = 0

        def reserve_call(self, **kwargs):
            self.reservation_reads += 1
            state = super().reserve_call(**kwargs)
            if self.refreshed and refresh_patch is not None:
                state.update(refresh_patch)
            return state

        def list_ledger(self, *, call_identity=None, **kwargs):
            if self.settle_during_read:
                self.settle_during_read = False
                self.refreshed = True
                self.calls[call_identity].update(
                    kind="settlement", actual=50_000,
                    terminal=br._terminal_response_document(
                        200 if successful else 502, {},
                        saved_body if successful else br.operations.GENERIC_UNAVAILABLE_BODY,
                        call_succeeded=successful),
                )
            return super().list_ledger(call_identity=call_identity, **kwargs)

    transport = ErrorTransport()
    transport.status = 502 if successful else 404
    first, broker, store, context, arguments = execute(transport, store=ConcurrentSettlementStore())
    store.settle_during_read = True
    replay = broker.execute(context, **arguments)
    assert store.reservation_reads == 3  # Initial dispatch, replay, one bounded refresh.
    assert len(transport.sent) == 2
    if refresh_patch is None:
        assert replay.status == (200 if successful else 404)
        assert replay.body == (saved_body if successful else first.body)
        assert replay.call["outcome"] == "settled" and replay.call["actual_microusd"] == 50_000
    else:
        assert replay.status == 503 and replay.call["error_code"] == "broker_unavailable"
        assert "actual_microusd" not in replay.call


def test_settlement_placeholder_with_success_flag_still_recovers_completed_response(monkeypatch):
    from tests.lab_arena.deepline_late_response_recovery_test import setup as recovery_setup
    from tests.lab_arena.deepline_completed_response_recovery_test import execute as recover, record

    broker, store, transport, context = recovery_setup(monkeypatch)
    first = recover(broker, context)
    assert first.status == 502
    original_list_ledger = store.list_ledger

    def settle_before_history_read(**kwargs):
        monkeypatch.setattr(store, "list_ledger", original_list_ledger)
        # Billing-only settlement can preserve an accepted-call success flag
        # while its terminal body is still the generic unavailable placeholder.
        store.calls[first.call["call_identity"]].update(
            kind="settlement", actual=2000,
            terminal=br._terminal_response_document(
                502, {"content-type": "application/json"},
                br.operations.GENERIC_UNAVAILABLE_BODY, call_succeeded=True),
        )
        return original_list_ledger(**kwargs)

    monkeypatch.setattr(store, "list_ledger", settle_before_history_read)
    transport.complete = transport.bill = True
    before = len(transport.requests)
    result = recover(broker, context)
    assert result.status == 200 and result.call["actual_microusd"] == 2000
    assert json.loads(result.body)["result"] == record()["response"]["result"]
    assert len(transport.requests) == before + 1
    assert transport.requests[-1]["method"] == "GET"
    assert transport.paid == 1 and sum(r["method"] == "POST" for r in transport.requests) == 1
