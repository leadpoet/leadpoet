"""Broker state machine, cost rules, and error genericness (labarena.md 7.3-7.5, 18.3, 18.4)."""

from __future__ import annotations

import base64
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from lab_arena import broker as br
from lab_arena import contracts, operations
from lab_arena.credentials import RuntimeKeyHandle

KEY = "sk-or-v1-" + "k" * 40
EXA_KEY = "exa-secret-" + "e" * 30
DOG_KEY = "dog-secret-" + "d" * 30


def price_table():
    return br.validate_price_table({
        "schema_version": br.PRICE_TABLE_SCHEMA_VERSION,
        "fetched_at": "2026-09-02T00:00:00Z",
        "source": br.OPENROUTER_MODELS_URL,
        "models": {
            "openai/gpt-4o-mini": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0", "image": "0", "web_search": "0", "internal_reasoning": "0"},
            "anthropic/claude-3.5-haiku": {"prompt": "0.0000008", "completion": "0.000004", "request": "0.00001", "image": "0", "web_search": "0", "internal_reasoning": "0.000004"},
        },
    })


class FakeLedgerStore:
    """In-memory model of the section 7.5 ledger functions and their statuses."""

    def __init__(self, *, per_icp_cap=1_000_000, balance=10_000_000, openrouter_capacity=10_000_000):
        self.per_icp_cap = per_icp_cap
        self.balance = balance
        self.openrouter_capacity = openrouter_capacity
        self.calls: Dict[str, Dict[str, Any]] = {}
        self.run = {"event_cursor": 0, "event_head_hash": ""}
        self.events: List[Dict[str, Any]] = []
        self.stale = False
        self.lock = threading.Lock()
        self.log: List[str] = []

    def _view(self, call):
        return {"status": {"reservation": "reserved", "dispatch": "dispatched", "settlement": "settled", "uncertain": "uncertain", "recovery": "recovered", "refusal": "refused"}[call["kind"]], "idempotent": True, "call_identity": call["identity"], "amount_microusd": call["amount"], "terminal_response": call.get("terminal"), "reason": call.get("reason"), "event_cursor": self.run["event_cursor"], "event_head_hash": self.run["event_head_hash"]}

    def _consumed(self):
        return sum(c["amount"] for c in self.calls.values() if c["kind"] in ("reservation", "dispatch", "settlement", "uncertain"))

    def reserve_call(self, *, run_id, lease_token_hash, call_identity, operation_id, provider, funding_source, amount_microusd, call_doc, lease_ttl_seconds):
        with self.lock:
            self.log.append("reserve")
            if self.stale:
                return {"status": "stale"}
            existing = self.calls.get(call_identity)
            if existing:
                return self._view(existing)
            reason = None
            if self._consumed() + amount_microusd > self.per_icp_cap:
                reason = "per_icp_cap"
            elif funding_source == "tao" and self.balance < amount_microusd:
                reason = "balance"
            elif funding_source == "openrouter" and self.openrouter_capacity < amount_microusd:
                reason = "key_capacity"
            if reason:
                self.calls[call_identity] = {"kind": "refusal", "identity": call_identity, "amount": 0, "reason": reason}
                return {"status": "refused", "idempotent": False, "reason": reason, "call_identity": call_identity, "event_cursor": self.run["event_cursor"], "event_head_hash": self.run["event_head_hash"]}
            if funding_source == "tao":
                self.balance -= amount_microusd
            else:
                self.openrouter_capacity -= amount_microusd
            self.calls[call_identity] = {"kind": "reservation", "identity": call_identity, "amount": amount_microusd, "funding": funding_source}
            return {"status": "reserved", "idempotent": False, "call_identity": call_identity, "amount_microusd": amount_microusd, "event_cursor": self.run["event_cursor"], "event_head_hash": self.run["event_head_hash"]}

    def mark_dispatched(self, *, run_id, lease_token_hash, call_identity):
        with self.lock:
            self.log.append("dispatch")
            if self.stale:
                return {"status": "stale"}
            call = self.calls.get(call_identity)
            if call is None:
                return {"status": "not_reserved"}
            if call["kind"] != "reservation":
                return self._view(call)
            call["kind"] = "dispatch"
            return {"status": "dispatched", "idempotent": False, "call_identity": call_identity, "amount_microusd": call["amount"], "event_cursor": self.run["event_cursor"], "event_head_hash": self.run["event_head_hash"]}

    def _append(self, event):
        if event["sequence"] != self.run["event_cursor"] or event["prev_hash"] != self.run["event_head_hash"]:
            from lab_arena.store import ArenaStoreError

            raise ArenaStoreError("rpc failed: lab_arena_event_sequence")
        self.events.append(event)
        self.run = {"event_cursor": event["sequence"] + 1, "event_head_hash": event["event_hash"]}

    def settle_call(self, *, run_id, lease_token_hash, call_identity, actual_microusd, terminal_response, event, lease_ttl_seconds):
        with self.lock:
            self.log.append("settle")
            if self.stale:
                return {"status": "stale"}
            call = self.calls[call_identity]
            if call["kind"] != "dispatch":
                return self._view(call)
            assert actual_microusd <= call["amount"]
            if call["funding"] == "tao":
                assert actual_microusd == call["amount"]
            # One transaction: the event check happens before any state changes.
            if event is not None:
                self._append(event)
            if call["funding"] != "tao":
                self.openrouter_capacity += call["amount"]  # outstanding released
                self.openrouter_capacity -= actual_microusd
            call.update({"kind": "settlement", "terminal": terminal_response, "actual": actual_microusd})
            return {"status": "settled", "idempotent": False, "actual_microusd": actual_microusd, "released_microusd": call["amount"] - actual_microusd, "terminal_response": terminal_response, "event_cursor": self.run["event_cursor"], "event_head_hash": self.run["event_head_hash"]}

    def mark_uncertain(self, *, run_id, lease_token_hash, call_identity, call_doc, event, lease_ttl_seconds):
        with self.lock:
            self.log.append("uncertain")
            if self.stale:
                return {"status": "stale"}
            call = self.calls[call_identity]
            if call["kind"] != "dispatch":
                return self._view(call)
            if event is not None:
                self._append(event)
            call["kind"] = "uncertain"
            return {"status": "uncertain", "idempotent": False, "amount_microusd": call["amount"], "event_cursor": self.run["event_cursor"], "event_head_hash": self.run["event_head_hash"]}

    def get_run(self, run_id):
        return dict(self.run)


class FakeTransport:
    def __init__(self, responses=None, *, fail=False):
        self.responses = list(responses or [])
        self.fail = fail
        self.sent: List[Dict[str, Any]] = []

    def send(self, *, method, url, headers, body, timeout_seconds):
        self.sent.append({"method": method, "url": url, "headers": dict(headers), "body": body, "timeout": timeout_seconds})
        if self.fail:
            raise br.ProviderTransportError("ReadTimeout")
        status, payload = self.responses.pop(0) if self.responses else (200, {"data": []})
        raw = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload
        return br.ProviderResponse(status, {"content-type": "application/json", "x-ratelimit-remaining": "3", "set-cookie": "s=1"}, raw)


def make_broker(store=None, transport=None, **kwargs):
    store = store or FakeLedgerStore()
    transport = transport or FakeTransport()
    broker = br.Broker(
        store=store,
        credentials=br.ArenaProviderCredentials(exa_api_key=EXA_KEY, scrapingdog_api_key=DOG_KEY),
        openrouter_key_for=lambda hotkey: RuntimeKeyHandle(KEY),
        price_table=price_table(),
        allowed_models=["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"],
        transport=transport,
        clock=lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc),
        **kwargs,
    )
    return broker, store, transport


CONTEXT = br.RunContext(run_id="r1", assignment_id="arena-2026-09-02:s1:1:0", icp_position=0, lease_token_hash=contracts.document_hash("lease"), miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", submission_id="s1", stage=1)
CHAT = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "find fintech companies"}], "max_tokens": 200}


def test_exa_call_reserves_dispatches_sends_and_settles_at_estimate():
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": [{"url": "https://a.example"}]})]))
    result = broker.execute(CONTEXT, operation_id="exa.search", parameters={"query": "fintech"}, action_sequence=0, timeout_ms=5000)
    assert result.status == 200 and json.loads(result.body)["results"][0]["url"] == "https://a.example"
    assert set(result.headers) == {"content-type", "content-length"}
    assert store.log == ["reserve", "dispatch", "settle"]
    call = result.call
    assert call["outcome"] == "settled" and call["reserved_microusd"] == call["actual_microusd"] == 5000
    sent = transport.sent[0]
    assert sent["url"] == "https://api.exa.ai/search" and sent["method"] == "POST"
    assert sent["headers"]["x-api-key"] == EXA_KEY and "authorization" not in sent["headers"]
    body = json.loads(sent["body"])
    assert body["numResults"] == 10 and body["contents"] == {"text": {"maxCharacters": 2000}} and body["query"] == "fintech"
    assert store.balance == 10_000_000 - 5000
    assert store.events[0]["event_type"] == "provider_call" and store.events[0]["payload"]["actual_microusd"] == 5000
    assert contracts.verify_event_chain(store.events) == store.run["event_head_hash"]


def test_scrapingdog_credential_goes_in_the_query_and_never_in_the_model_response():
    broker, store, transport = make_broker(transport=FakeTransport([(200, b"<html>hi</html>")]))
    result = broker.execute(CONTEXT, operation_id="scrapingdog.scrape", parameters={"url": "https://example.com/about"}, action_sequence=0, timeout_ms=5000)
    assert result.status == 200
    sent = transport.sent[0]
    assert "api_key=" + DOG_KEY in sent["url"] and sent["url"].startswith("https://api.scrapingdog.com/scrape?")
    assert "premium=false" in sent["url"]
    assert DOG_KEY not in result.body.decode() and DOG_KEY not in json.dumps(result.call)
    assert result.call["reserved_microusd"] == 2000


def test_openrouter_reserves_maximum_cost_and_settles_actual_from_pinned_table():
    usage = {"prompt_tokens": 20, "completion_tokens": 50}
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"id": "gen", "model": "openai/gpt-4o-mini", "choices": [], "usage": usage})]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.status == 200
    expected_max = br.max_openrouter_cost_microusd(price_table(), "openai/gpt-4o-mini", dict(CHAT), max_output_tokens=200)
    expected_actual = br.actual_openrouter_cost_microusd(price_table(), "openai/gpt-4o-mini", {"usage": usage})
    assert result.call["reserved_microusd"] == expected_max and result.call["actual_microusd"] == expected_actual < expected_max
    sent = transport.sent[0]
    assert sent["headers"]["authorization"] == "Bearer " + KEY
    body = json.loads(sent["body"])
    assert body["provider"] == {"allow_fallbacks": False, "data_collection": "deny"} and body["stream"] is False
    assert store.openrouter_capacity == 10_000_000 - expected_actual


@pytest.mark.parametrize("payload", [
    {"choices": []},
    {"usage": {"prompt_tokens": "x", "completion_tokens": 1}},
    {"usage": {"prompt_tokens": -1, "completion_tokens": 1}},
    {"model": "other/model", "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    {"usage": {"prompt_tokens": 10 ** 9, "completion_tokens": 10 ** 9}},
])
def test_missing_malformed_stale_or_excessive_usage_never_settles_below_the_reservation(payload):
    broker, store, transport = make_broker(transport=FakeTransport([(200, payload)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=30000)
    assert result.call["actual_microusd"] == result.call["reserved_microusd"] > 0
    assert store.calls[result.call["call_identity"]]["actual"] == result.call["reserved_microusd"]


def test_max_tokens_is_capped_and_disallowed_models_are_refused_before_reservation():
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})]))
    huge = dict(CHAT, max_tokens=4096)
    broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=huge, action_sequence=0, timeout_ms=1000)
    assert json.loads(transport.sent[0]["body"])["max_tokens"] == operations.OPENROUTER_MAX_OUTPUT_TOKENS
    other = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=dict(CHAT, model="openai/gpt-5"), action_sequence=1, timeout_ms=1000)
    assert other.status == 400 and json.loads(other.body) == {"error": {"code": "model_not_allowed"}}
    assert store.log.count("reserve") == 1
    bad = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=dict(CHAT, tools=[{"x": 1}]), action_sequence=2, timeout_ms=1000)
    assert bad.status == 400 and json.loads(bad.body) == {"error": {"code": "invalid_request"}}
    unknown = broker.execute(CONTEXT, operation_id="deepline.play", parameters={}, action_sequence=3, timeout_ms=1000)
    assert unknown.status == 400


def test_budget_refusal_is_generic_and_recorded_under_the_identity():
    broker, store, transport = make_broker(store=FakeLedgerStore(per_icp_cap=4000))
    refused = broker.execute(CONTEXT, operation_id="exa.search", parameters={"query": "x"}, action_sequence=0, timeout_ms=1000)
    assert refused.status == 402 and json.loads(refused.body) == {"error": {"code": "budget_refused"}}
    assert refused.call["outcome"] == "refused" and refused.call["reason"] == "per_icp_cap"
    assert transport.sent == []
    again = broker.execute(CONTEXT, operation_id="exa.search", parameters={"query": "x"}, action_sequence=0, timeout_ms=1000)
    assert again.status == 402 and store.log == ["reserve", "reserve"]


def test_transport_failure_after_send_marks_uncertain_and_keeps_full_reservation():
    broker, store, transport = make_broker(transport=FakeTransport(fail=True))
    result = broker.execute(CONTEXT, operation_id="exa.search", parameters={"query": "x"}, action_sequence=0, timeout_ms=1000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert store.calls[result.call["call_identity"]]["kind"] == "uncertain"
    assert store.balance == 10_000_000 - 5000
    assert result.call["outcome"] == "uncertain" and result.call["actual_microusd"] == 5000
    # A later identical request neither re-sends nor releases the reservation.
    late = broker.execute(CONTEXT, operation_id="exa.search", parameters={"query": "x"}, action_sequence=0, timeout_ms=1000)
    assert late.status == 409 and json.loads(late.body) == {"error": {"code": "call_uncertain"}} and len(transport.sent) == 1


def test_fault_injection_points_produce_single_accounting_results():
    # After reservation (crash before dispatch): resuming the identity dispatches exactly once.
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": []})]))
    identity_args = dict(operation_id="exa.search", parameters={"query": "crash"}, action_sequence=5, timeout_ms=1000)
    request_hash = contracts.document_hash(operations.validate_operation_request("exa.search", {"query": "crash"}))
    identity = contracts.provider_call_identity(assignment_id=CONTEXT.assignment_id, icp_position=0, action_sequence=5, operation_id="exa.search", request_hash=request_hash)
    store.reserve_call(run_id="r1", lease_token_hash=CONTEXT.lease_token_hash, call_identity=identity, operation_id="exa.search", provider="exa", funding_source="tao", amount_microusd=5000, call_doc={}, lease_ttl_seconds=420)
    result = broker.execute(CONTEXT, **identity_args)
    assert result.status == 200 and len(transport.sent) == 1
    assert [c for c in store.calls.values() if c["kind"] == "settlement"] and store.balance == 10_000_000 - 5000
    # After the dispatch marker (crash before send): the repeat never sends and reports uncertain.
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": []})]))
    store.reserve_call(run_id="r1", lease_token_hash=CONTEXT.lease_token_hash, call_identity=identity, operation_id="exa.search", provider="exa", funding_source="tao", amount_microusd=5000, call_doc={}, lease_ttl_seconds=420)
    store.mark_dispatched(run_id="r1", lease_token_hash=CONTEXT.lease_token_hash, call_identity=identity)
    result = broker.execute(CONTEXT, **identity_args)
    assert result.status == 409 and transport.sent == []
    # After settlement / HTTP response loss: the repeat returns the stored terminal response without a send.
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": [1]})]))
    first = broker.execute(CONTEXT, **identity_args)
    second = broker.execute(CONTEXT, **identity_args)
    assert second.status == first.status and second.body == first.body and second.call["idempotent"] is True
    assert len(transport.sent) == 1 and store.log.count("settle") == 1
    # Stage closed between reservation and dispatch: the marker fails and nothing is sent.
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": []})]))
    original = store.mark_dispatched

    def closing(**kwargs):
        store.stale = True
        return original(**kwargs)

    store.mark_dispatched = closing
    result = broker.execute(CONTEXT, **identity_args)
    assert result.status == 409 and json.loads(result.body) == {"error": {"code": "lease_stale"}} and transport.sent == []


def test_two_broker_instances_cause_at_most_one_dispatch_per_identity():
    store = FakeLedgerStore()
    responses = [(200, {"results": ["a"]}), (200, {"results": ["b"]})]
    transports = [FakeTransport([responses[0]]), FakeTransport([responses[1]])]
    brokers = [make_broker(store=store, transport=transports[i])[0] for i in range(2)]
    results = []
    barrier = threading.Barrier(2)

    def worker(index):
        barrier.wait(timeout=10)
        results.append(brokers[index].execute(CONTEXT, operation_id="exa.search", parameters={"query": "race"}, action_sequence=9, timeout_ms=1000))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    sends = len(transports[0].sent) + len(transports[1].sent)
    assert sends == 1
    settled = [r for r in results if r.status == 200]
    assert len(settled) >= 1
    assert [c for c in store.calls.values() if c["kind"] == "settlement"]


def test_event_cursor_race_is_retried_with_a_fresh_cursor():
    broker, store, transport = make_broker(transport=FakeTransport([(200, {"results": []})]))
    # A worker event lands between dispatch and settle.
    original = store.mark_dispatched

    def dispatch_then_append(**kwargs):
        response = original(**kwargs)
        event = contracts.build_private_event(event_type="stdout", sequence=store.run["event_cursor"], prev_hash=store.run["event_head_hash"], timestamp="2026-09-02T01:00:00Z", payload={"line": "x"})
        store._append(event)
        return response

    store.mark_dispatched = dispatch_then_append
    result = broker.execute(CONTEXT, operation_id="exa.search", parameters={"query": "x"}, action_sequence=0, timeout_ms=1000)
    assert result.status == 200 and [e["event_type"] for e in store.events] == ["stdout", "provider_call"]
    assert contracts.verify_event_chain(store.events)


def test_errors_and_responses_carry_no_provider_account_or_credential_detail():
    broker, store, transport = make_broker(transport=FakeTransport([(401, {"error": {"message": "invalid api key " + EXA_KEY}})]))
    result = broker.execute(CONTEXT, operation_id="exa.search", parameters={"query": "x"}, action_sequence=0, timeout_ms=1000)
    assert result.status == 502 and json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    assert EXA_KEY not in result.body.decode() and "x-ratelimit-remaining" not in result.headers and "set-cookie" not in result.headers
    document = result.to_document()
    assert set(document) == {"status", "headers", "body_b64", "call"}
    for value in (json.dumps(document), repr(broker._credentials)):
        assert EXA_KEY not in value and DOG_KEY not in value and KEY not in value


def test_price_table_parsing_and_validation():
    response = {"data": [
        {"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0", "image": "0", "web_search": "0", "internal_reasoning": "0"}},
        {"id": "other/model", "pricing": {"prompt": "1"}},
    ]}
    table = br.price_table_from_models_response(response, ["openai/gpt-4o-mini"], fetched_at="2026-09-02T00:00:00Z")
    assert table["models"]["openai/gpt-4o-mini"]["completion"] == "0.0000006" and table["price_table_hash"].startswith("sha256:")
    assert br.validate_price_table(table)["price_table_hash"] == table["price_table_hash"]
    with pytest.raises(contracts.ArenaContractError):
        br.price_table_from_models_response(response, ["missing/model"], fetched_at="2026-09-02T00:00:00Z")
    with pytest.raises(contracts.ArenaContractError):
        br.validate_price_table(dict(table, models={}))
    with pytest.raises(contracts.ArenaContractError):
        br.validate_price_table(dict(table, price_table_hash=contracts.document_hash("x")))
    with pytest.raises(contracts.ArenaContractError):
        br.Broker(store=FakeLedgerStore(), credentials=br.ArenaProviderCredentials("a", "b"), openrouter_key_for=lambda h: None, price_table=table, allowed_models=["anthropic/claude-3.5-haiku"], transport=FakeTransport())
    cost = br.max_openrouter_cost_microusd(price_table(), "anthropic/claude-3.5-haiku", {"messages": [{"role": "user", "content": "hi"}]}, max_output_tokens=100)
    # 26 bounded input tokens * 0.8e-6 + 100 * (4e-6 + 4e-6 reasoning) + 1e-5 request = 0.0008308 USD -> 831 micro-USD (ceiling)
    assert cost == 831
    parsed = br.parse_broker_document({"status": 200, "headers": {"content-type": "application/json"}, "body_b64": base64.b64encode(b"{}").decode(), "call": {"a": 1}})
    assert parsed.body == b"{}" and parsed.call == {"a": 1}
