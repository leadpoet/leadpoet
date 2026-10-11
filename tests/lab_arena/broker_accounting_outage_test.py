"""A database restart must not discard the broker's exact paid-call receipt."""
from copy import deepcopy
import json

import pytest

from lab_arena import broker as br
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, CHAT, FakeLedgerStore, FakeTransport, make_broker,
)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


def clock_fixture(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(br.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(br.time, "sleep", clock.sleep)
    return clock


@pytest.mark.parametrize("failure", [
    br.ArenaStoreUnavailable("transport unavailable"),
    br.ArenaStoreError("rpc lab_arena_settle_call failed: HTTP 503 Service Unavailable"),
    br.ArenaStoreError("rpc lab_arena_settle_call returned non-JSON"),
])
@pytest.mark.parametrize("lost_reply", [False, True])
def test_exact_paid_reply_survives_100_second_database_outage(monkeypatch, failure, lost_reply):
    clock = clock_fixture(monkeypatch)
    broker, store, transport = make_broker(transport=FakeTransport([(200, {
        "model": "openai/gpt-4o-mini", "choices": [],
        "usage": {"cost": "0.0000123"},
    })]))
    write = store.settle_call
    attempts = []

    def unavailable(**kwargs):
        attempts.append(deepcopy(kwargs))
        if lost_reply and len(attempts) == 1:
            write(**kwargs)
        if clock.now < 100:
            raise failure
        return write(**kwargs)

    store.settle_call = unavailable
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
                            action_sequence=0, timeout_ms=5000)
    assert result.status == 200 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 13
    assert 100 <= clock.now <= 120
    assert all(attempt == attempts[0] for attempt in attempts)
    assert len(transport.sent) == 1
    assert next(iter(store.calls.values()))["actual"] == 13
    assert "uncertain" not in store.log
    assert store.openrouter_capacity == 10_000_000 - 13


@pytest.mark.parametrize("transport_failure", [False, True])
def test_unknown_receipt_survives_outage_without_inventing_cost(monkeypatch, transport_failure):
    clock = clock_fixture(monkeypatch)
    broker, store, transport = make_broker(transport=FakeTransport(
        [(200, b'{"job_id":"job-unknown","status":"completed","results":[]}')],
        fail=transport_failure,
    ))
    write = store.mark_uncertain
    attempts = []

    def unavailable(**kwargs):
        attempts.append(deepcopy(kwargs))
        if clock.now < 100:
            raise br.ArenaStoreError("rpc lab_arena_mark_uncertain failed: HTTP 503 Service Unavailable")
        return write(**kwargs)

    store.mark_uncertain = unavailable
    result = broker.execute(CONTEXT, operation_id="deepline.execute",
                            parameters={"tool": "exa_search", "payload": {"query": "x"}},
                            action_sequence=0, timeout_ms=5000)
    assert result.call["outcome"] == "uncertain"
    call = store.calls[result.call["call_identity"]]
    assert call["kind"] == "uncertain" and "actual" not in call
    assert all(attempt == attempts[0] for attempt in attempts)
    assert call["uncertain_doc"]["reason"] == (
        "transport_failure" if transport_failure else "missing_provider_cost"
    )
    assert call["uncertain_doc"]["deepline_request_id"] == transport.sent[0]["headers"]["x-deepline-request-id"]
    assert [request["method"] for request in transport.sent].count("POST") == 1


def test_bounded_outage_does_not_falsely_report_a_durable_uncertain_head(monkeypatch):
    clock = clock_fixture(monkeypatch)
    broker, store, transport = make_broker(transport=FakeTransport([(200, {
        "model": "openai/gpt-4o-mini", "choices": [], "usage": {"cost": "0.0000123"},
    })]))

    def unavailable(**kwargs):
        raise br.ArenaStoreUnavailable("transport unavailable")

    store.settle_call = unavailable
    store.mark_uncertain = unavailable
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
                            action_sequence=0, timeout_ms=5000)
    assert clock.now == 120
    assert result.call["outcome"] == "dispatched"
    assert json.loads(result.body)["error"]["code"] == "broker_unavailable"
    assert next(iter(store.calls.values()))["kind"] == "dispatch"
    assert len(transport.sent) == 1


@pytest.mark.parametrize("failure", [
    br.ArenaStoreError("rpc lab_arena_settle_call failed: HTTP 403 forbidden"),
    br.ArenaStoreError("rpc lab_arena_settle_call failed: HTTP 400 bad input"),
    br.ArenaStoreError("rpc lab_arena_reserve_call failed: HTTP 503 unavailable"),
    br.ArenaStoreError("settlement contract mismatch"),
])
def test_nonavailability_errors_keep_existing_immediate_failure_path(monkeypatch, failure):
    clock = clock_fixture(monkeypatch)
    broker, store, transport = make_broker(transport=FakeTransport([(200, {
        "model": "openai/gpt-4o-mini", "choices": [], "usage": {"cost": "0.0000123"},
    })]))
    attempts = []

    def rejected(**kwargs):
        attempts.append(kwargs)
        raise failure

    store.settle_call = rejected
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT,
                            action_sequence=0, timeout_ms=5000)
    assert result.call["outcome"] == "uncertain"
    assert len(attempts) == br._SETTLEMENT_STORE_MAX_ATTEMPTS
    assert clock.sleeps == []
    assert len(transport.sent) == 1
    assert next(iter(store.calls.values()))["uncertain_doc"]["known_actual_microusd"] == 13
