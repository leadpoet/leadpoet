"""Shared OpenRouter Responses admission across concurrent Arena rounds."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest

from lab_arena import broker as br
from lab_arena import operations
from lab_arena import runner as rn
from lab_arena.service import ServiceError
from lab_arena.wiring import _openrouter_shared_concurrency_from_environment
from tests.lab_arena.codex_runtime_test import response
from tests.lab_arena.test_lab_arena_broker import (
    CHAT,
    CONTEXT,
    FakeLedgerStore,
    FakeTransport,
    HOST_KEYS,
    make_broker,
)
from tests.lab_arena.test_lab_arena_runner import lease


PARAMETERS = {"model": "openai/gpt-4o-mini", "input": "research"}


def fingerprint(secret: str = HOST_KEYS["openrouter"]) -> str:
    return br._credential_fingerprint(secret)


def context(number: int):
    return replace(
        CONTEXT,
        run_id="run-gate-%d" % number,
        assignment_id="assignment-gate-%d" % number,
        round_id="arena-gate-%d" % number,
    )


def wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached")
        time.sleep(0.002)


def queued(gate: br.OpenRouterSharedGate, key: str) -> int:
    with gate._condition:
        state = gate._states.get(key)
        return 0 if state is None else len(state.waiters)


def test_ten_callers_are_bounded_and_fifo_without_starvation():
    gate = br.OpenRouterSharedGate(max_concurrency=2)
    key = fingerprint()
    first = gate.acquire(key, deadline=time.monotonic() + 2)
    second = gate.acquire(key, deadline=time.monotonic() + 2)
    assert first is not None and second is not None

    leases = {}
    active = 0
    maximum_active = 0
    lock = threading.Lock()
    release = [threading.Event() for _ in range(8)]

    def caller(index: int) -> None:
        nonlocal active, maximum_active
        lease = gate.acquire(key, deadline=time.monotonic() + 2)
        assert lease is not None
        with lock:
            leases[index] = lease
            active += 1
            maximum_active = max(maximum_active, active)
        assert release[index].wait(2)
        with lock:
            active -= 1
        lease.release()

    threads = []
    for index in range(8):
        thread = threading.Thread(target=caller, args=(index,))
        thread.start()
        threads.append(thread)
        wait_for(lambda index=index: queued(gate, key) == index + 1)

    first.release()
    second.release()
    for index in range(8):
        wait_for(lambda index=index: index in leases)
        release[index].set()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()

    assert [leases[index].ticket for index in range(8)] == sorted(
        lease.ticket for lease in leases.values()
    )
    assert maximum_active <= 2


def test_capacity_wait_blocks_on_condition_instead_of_spinning():
    class RecordingCondition(threading.Condition):
        def __init__(self):
            super().__init__()
            self.first_wait = threading.Event()
            self.timeouts = []

        def wait(self, timeout=None):
            self.timeouts.append(timeout)
            self.first_wait.set()
            return super().wait(timeout)

    gate = br.OpenRouterSharedGate(max_concurrency=1)
    condition = RecordingCondition()
    gate._condition = condition
    held = gate.acquire(fingerprint(), deadline=time.monotonic() + 2)
    assert held is not None
    acquired = []
    waiter = threading.Thread(
        target=lambda: acquired.append(
            gate.acquire(fingerprint(), deadline=time.monotonic() + 1)
        )
    )
    waiter.start()
    assert condition.first_wait.wait(1)
    assert condition.timeouts[0] > 0.5
    held.release()
    waiter.join(2)
    assert acquired[0] is not None
    acquired[0].release()


class FirstCallBlockingTransport:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.calls = 0
        self.sent = []

    def send(self, *, method, url, headers, body, timeout_seconds, **_kwargs):
        with self.lock:
            self.calls += 1
            call = self.calls
            self.sent.append((call, timeout_seconds))
        if call == 1:
            self.entered.set()
            assert self.release.wait(2)
        return br.ProviderResponse(
            200,
            {"content-type": "application/json"},
            json.dumps(response(id="gen-%d" % call)).encode(),
        )


def test_one_gate_is_shared_across_round_brokers_before_any_reservation():
    gate = br.OpenRouterSharedGate(max_concurrency=1)
    transport = FirstCallBlockingTransport()
    first, first_store, _ = make_broker(
        store=FakeLedgerStore(), transport=transport, openrouter_shared_gate=gate
    )
    second, second_store, _ = make_broker(
        store=FakeLedgerStore(), transport=transport, openrouter_shared_gate=gate
    )
    results = {}

    thread1 = threading.Thread(
        target=lambda: results.setdefault(
            1,
            first.execute(
                context(1), operation_id="openrouter.responses",
                parameters=PARAMETERS, action_sequence=1, timeout_ms=120_000,
            ),
        )
    )
    thread2 = threading.Thread(
        target=lambda: results.setdefault(
            2,
            second.execute(
                context(2), operation_id="openrouter.responses",
                parameters=PARAMETERS, action_sequence=1, timeout_ms=120_000,
            ),
        )
    )
    thread1.start()
    assert transport.entered.wait(2)
    thread2.start()
    wait_for(lambda: queued(gate, fingerprint()) == 1)
    assert second_store.log == []
    transport.release.set()
    thread1.join(2)
    thread2.join(2)

    assert not thread1.is_alive() and not thread2.is_alive()
    assert results[1].status == results[2].status == 200
    assert first_store.log == second_store.log == ["reserve", "dispatch", "settle"]


def test_queued_cancellation_removes_waiter_without_ledger_or_provider_call():
    gate = br.OpenRouterSharedGate(max_concurrency=1)
    held = gate.acquire(fingerprint(), deadline=time.monotonic() + 2)
    assert held is not None
    cancelled = threading.Event()
    broker, store, transport = make_broker(openrouter_shared_gate=gate)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            broker.execute(
                context(3), operation_id="openrouter.responses",
                parameters=PARAMETERS, action_sequence=1, timeout_ms=120_000,
                cancel_requested=cancelled.is_set,
            )
        )
    )
    thread.start()
    wait_for(lambda: queued(gate, fingerprint()) == 1)
    cancelled.set()
    thread.join(2)
    held.release()

    assert not thread.is_alive()
    assert result[0].call["outcome"] == "not_dispatched"
    assert result[0].call["reason"] == "provider_admission_cancelled"
    assert "provider_status" not in result[0].call
    assert store.log == [] and transport.sent == []
    assert queued(gate, fingerprint()) == 0


def test_miner_key_uses_its_credential_gate_and_missing_key_never_retries():
    miner_key = "sk-or-v1-" + "m" * 40
    miner_fingerprint = fingerprint(miner_key)
    gate = br.OpenRouterSharedGate(max_concurrency=1)
    held = gate.acquire(miner_fingerprint, deadline=time.monotonic() + 2)
    assert held is not None
    cancelled = threading.Event()
    broker, store, transport = make_broker(
        openrouter_shared_gate=gate,
        credential_for=lambda _context, _provider: miner_key,
        provider_funding_source_for=lambda _context, _provider: "miner_key",
    )
    results = []
    thread = threading.Thread(
        target=lambda: results.append(
            broker.execute(
                context(31), operation_id="openrouter.responses",
                parameters=PARAMETERS, action_sequence=1, timeout_ms=120_000,
                cancel_requested=cancelled.is_set,
            )
        )
    )
    thread.start()
    wait_for(lambda: queued(gate, miner_fingerprint) == 1)
    cancelled.set()
    thread.join(2)
    held.release()
    assert results[0].call["funding_source"] == "miner_key"
    assert store.log == [] and transport.sent == []

    def unavailable(_context, _provider):
        raise br.BrokerError("miner_credentials_unavailable")

    missing, missing_store, missing_transport = make_broker(
        openrouter_shared_gate=br.OpenRouterSharedGate(1),
        credential_for=unavailable,
        provider_funding_source_for=lambda _context, _provider: "miner_key",
    )
    refused = missing.execute(
        context(32), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=120_000,
    )
    assert refused.call["error_code"] == "miner_credentials_unavailable"
    assert missing_store.log == [] and missing_transport.sent == []


def test_different_credentials_are_independent_and_idle_state_is_bounded():
    gate = br.OpenRouterSharedGate(
        max_concurrency=1, idle_seconds=0, max_idle_states=1, max_states=2
    )
    key1 = fingerprint("sk-or-v1-" + "a" * 40)
    key2 = fingerprint("sk-or-v1-" + "b" * 40)
    key3 = fingerprint("sk-or-v1-" + "c" * 40)
    lease1 = gate.acquire(key1, deadline=time.monotonic() + 1)
    lease2 = gate.acquire(key2, deadline=time.monotonic() + 1)
    assert lease1 is not None and lease2 is not None
    assert gate.acquire(key3, deadline=time.monotonic() + 0.01) is None
    lease1.release()
    lease2.release()
    lease3 = gate.acquire(key3, deadline=time.monotonic() + 1)
    assert lease3 is not None
    lease3.release()
    assert len(gate._states) <= 2


def test_cooldown_uses_strict_settled_zero_429_and_stale_success_cannot_reset_it():
    gate = br.OpenRouterSharedGate(
        max_concurrency=2, fallback_seconds=(0.04, 0.08)
    )
    key = fingerprint()
    stale_success = gate.acquire(key, deadline=time.monotonic() + 1)
    throttle = gate.acquire(key, deadline=time.monotonic() + 1)
    assert stale_success is not None and throttle is not None
    assert gate.observe_throttle(throttle, br._RETRY_AFTER_ABSENT)
    gate.observe_success(stale_success)
    with gate._condition:
        state = gate._states[key]
        assert state.cooldown_generation == 1
        assert state.throttle_count == 1
        remaining = state.cooldown_until - time.monotonic()
    stale_success.release()
    throttle.release()
    started = time.monotonic()
    admitted = gate.acquire(key, deadline=time.monotonic() + 1)
    assert admitted is not None
    assert time.monotonic() - started >= max(0.0, remaining - 0.015)
    admitted.release()

    invalid = gate.acquire(key, deadline=time.monotonic() + 1)
    assert invalid is not None
    assert not gate.observe_throttle(invalid, None)
    invalid.release()


def test_strict_openrouter_429_cools_other_round_but_invalid_hint_does_not():
    rate_limit = {"error": {"code": "rate_limit_exceeded", "message": "limited"}}
    gate = br.OpenRouterSharedGate(
        max_concurrency=1, fallback_seconds=(0.04, 0.08)
    )
    first, _, _ = make_broker(
        transport=FakeTransport([(200, rate_limit)]),
        openrouter_shared_gate=gate,
    )
    limited = first.execute(
        context(4), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=120_000,
    )
    assert limited.status == 502
    assert limited.call["provider"] == "openrouter"
    assert limited.call["provider_status"] == 429
    assert limited.call["outcome"] == "settled"
    assert limited.call["actual_microusd"] == 0
    assert limited.call.get("idempotent", False) is False

    second, _, _ = make_broker(
        transport=FakeTransport([(200, response(id="gen-after-cooldown"))]),
        openrouter_shared_gate=gate,
    )
    started = time.monotonic()
    assert second.execute(
        context(5), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=120_000,
    ).status == 200
    assert time.monotonic() - started >= 0.025

    invalid_gate = br.OpenRouterSharedGate(
        max_concurrency=1, fallback_seconds=(1.0, 1.0)
    )
    invalid, _, _ = make_broker(
        transport=FakeTransport([(200, rate_limit, {"retry-after": "invalid"})]),
        openrouter_shared_gate=invalid_gate,
    )
    result = invalid.execute(
        context(6), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=120_000,
    )
    assert result.call["retry_after_seconds"] is None
    with invalid_gate._condition:
        assert invalid_gate._states[fingerprint()].cooldown_generation == 0


def test_retained_responses_429_pauses_gate_before_billing_readback(
    monkeypatch,
):
    retained_failure = {
        "id": "resp-retained-rate-limit",
        "object": "response",
        "status": "failed",
        "error": {
            "code": "rate_limit_exceeded",
            "message": "sanitized upstream throttle",
        },
        "output": [],
        "usage": None,
    }
    gate = br.OpenRouterSharedGate(
        max_concurrency=2, fallback_seconds=(1.0, 1.0)
    )
    held = gate.acquire(fingerprint(), deadline=time.monotonic() + 2)
    assert held is not None
    readback_entered = threading.Event()
    release_readback = threading.Event()

    def blocked_readback(**_kwargs):
        readback_entered.set()
        assert release_readback.wait(2)
        return None

    monkeypatch.setattr(br, "_openrouter_generation_readback", blocked_readback)
    limited, limited_store, limited_transport = make_broker(
        store=FakeLedgerStore(),
        transport=FakeTransport([(200, retained_failure)]),
        openrouter_shared_gate=gate,
    )
    following, following_store, following_transport = make_broker(
        store=FakeLedgerStore(),
        transport=FakeTransport([(200, response(id="gen-after-readback"))]),
        openrouter_shared_gate=gate,
    )
    results = {}
    limited_thread = threading.Thread(
        target=lambda: results.setdefault(
            "limited",
            limited.execute(
                context(41), operation_id="openrouter.responses",
                parameters=PARAMETERS, action_sequence=1, timeout_ms=120_000,
            ),
        )
    )
    limited_thread.start()
    assert readback_entered.wait(2)
    with gate._condition:
        state = gate._states[fingerprint()]
        assert state.cooldown_generation == 1
        assert state.throttle_count == 1

    cancelled = threading.Event()
    following_thread = threading.Thread(
        target=lambda: results.setdefault(
            "following",
            following.execute(
                context(42), operation_id="openrouter.responses",
                parameters=PARAMETERS, action_sequence=1, timeout_ms=120_000,
                cancel_requested=cancelled.is_set,
            ),
        )
    )
    following_thread.start()
    wait_for(lambda: queued(gate, fingerprint()) == 1)
    held.release()
    time.sleep(0.02)
    assert queued(gate, fingerprint()) == 1
    assert following_store.log == [] and following_transport.sent == []

    cancelled.set()
    following_thread.join(2)
    release_readback.set()
    limited_thread.join(2)

    assert not limited_thread.is_alive() and not following_thread.is_alive()
    assert results["following"].call["outcome"] == "not_dispatched"
    assert results["following"].call["reason"] == "provider_admission_cancelled"
    assert results["limited"].status == 502
    assert results["limited"].call["provider_status"] == 429
    assert results["limited"].call["outcome"] == "settled"
    assert results["limited"].call["actual_microusd"] == 0
    assert limited_store.log == ["reserve", "dispatch", "settle"]
    assert len(limited_transport.sent) == 1
    with gate._condition:
        state = gate._states[fingerprint()]
        assert state.cooldown_generation == 1
        assert state.throttle_count == 1


def test_early_429_pause_does_not_authorize_billed_worker_retry():
    billed_failure = {
        "object": "response",
        "status": "failed",
        "error": {"code": "rate_limit_exceeded", "message": "limited"},
        "output": [],
        "usage": {"cost": "0.000001"},
    }
    gate = br.OpenRouterSharedGate(
        max_concurrency=2, fallback_seconds=(0.01, 0.02)
    )
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, billed_failure)]),
        openrouter_shared_gate=gate,
    )

    result = broker.execute(
        context(43), operation_id="openrouter.responses",
        parameters=PARAMETERS, action_sequence=1, timeout_ms=120_000,
    )

    assert result.call["provider_status"] == 429
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 1
    assert rn.WorkerSocketServer._responses_rate_limit_delay(result.call, 0) is None
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1
    with gate._condition:
        state = gate._states[fingerprint()]
        assert state.cooldown_generation == 1
        assert state.throttle_count == 1


def test_early_429_pause_does_not_authorize_uncertain_worker_retry():
    free_failure = {
        "object": "response",
        "status": "failed",
        "error": {"code": "rate_limit_exceeded", "message": "limited"},
        "output": [],
        "usage": None,
    }
    gate = br.OpenRouterSharedGate(
        max_concurrency=2, fallback_seconds=(0.01, 0.02)
    )
    store = FakeLedgerStore()

    def fail_settlement(**_kwargs):
        store.log.append("settle")
        raise br.ArenaStoreError("synthetic settlement failure")

    store.settle_call = fail_settlement
    broker, _, transport = make_broker(
        store=store,
        transport=FakeTransport([(200, free_failure)]),
        openrouter_shared_gate=gate,
    )

    result = broker.execute(
        context(44), operation_id="openrouter.responses",
        parameters=PARAMETERS, action_sequence=1, timeout_ms=120_000,
    )

    assert result.call["outcome"] == "uncertain"
    assert rn.WorkerSocketServer._responses_rate_limit_delay(result.call, 0) is None
    assert store.log[:2] == ["reserve", "dispatch"]
    assert store.log[-1] == "uncertain"
    assert len(transport.sent) == 1
    assert next(iter(store.calls.values()))["kind"] == "uncertain"
    with gate._condition:
        state = gate._states[fingerprint()]
        assert state.cooldown_generation == 1
        assert state.throttle_count == 1


def test_reservation_boundary_oversleep_keeps_dispatch_and_settlement(monkeypatch):
    class SlowStore(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            time.sleep(0.06)
            return result

    monkeypatch.setattr(operations, "BUDGET_ADMISSION_MAX_SECONDS", 0.01)
    monkeypatch.setattr(operations, "PROVIDER_BILLING_RECONCILIATION_SECONDS", 0.01)
    monkeypatch.setattr(operations, "PROVIDER_API_TIMEOUT_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(br, "OPENROUTER_SHARED_GATE_MIN_PROVIDER_SECONDS", 0.01)
    broker, store, transport = make_broker(
        store=SlowStore(),
        transport=FakeTransport([(200, response(id="gen-slow-reserve"))]),
        openrouter_shared_gate=br.OpenRouterSharedGate(1),
    )

    result = broker.execute(
        context(7), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=50,
    )

    assert result.status == 200
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 12
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1
    assert next(iter(store.calls.values()))["kind"] == "settlement"


def test_disconnect_during_reserve_keeps_one_dispatch_and_settlement():
    cancelled = threading.Event()

    class CancellingStore(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            cancelled.set()
            return result

    broker, store, transport = make_broker(
        store=CancellingStore(),
        transport=FakeTransport([(200, response(id="gen-cancel-boundary"))]),
        openrouter_shared_gate=br.OpenRouterSharedGate(1),
    )
    result = broker.execute(
        context(9), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=120_000,
        cancel_requested=cancelled.is_set,
    )

    assert result.status == 200
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 12
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1
    settlement = next(iter(store.calls.values()))
    assert settlement["kind"] == "settlement"
    assert settlement["actual"] == 12


def test_slow_dispatch_marker_keeps_one_transport_and_settlement(monkeypatch):
    class SlowDispatchStore(FakeLedgerStore):
        def mark_dispatched(self, **kwargs):
            result = super().mark_dispatched(**kwargs)
            time.sleep(0.06)
            return result

    monkeypatch.setattr(operations, "BUDGET_ADMISSION_MAX_SECONDS", 0.01)
    monkeypatch.setattr(operations, "PROVIDER_BILLING_RECONCILIATION_SECONDS", 0.01)
    monkeypatch.setattr(operations, "PROVIDER_API_TIMEOUT_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(br, "OPENROUTER_SHARED_GATE_MIN_PROVIDER_SECONDS", 0.01)
    broker, store, transport = make_broker(
        store=SlowDispatchStore(),
        transport=FakeTransport([(200, response(id="gen-slow-dispatch"))]),
        openrouter_shared_gate=br.OpenRouterSharedGate(1),
    )

    result = broker.execute(
        context(8), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=50,
    )

    assert result.status == 200
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 12
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1


def test_idle_gate_preserves_short_responses_timeout():
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, response(id="gen-short-timeout"))]),
        openrouter_shared_gate=br.OpenRouterSharedGate(1)
    )

    result = broker.execute(
        context(10), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=25,
    )

    assert result.status == 200
    assert result.call["outcome"] == "settled"
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1
    assert 0 < transport.sent[0]["timeout"] <= 0.025


def test_nongated_chat_keeps_full_post_admission_provider_timeout(monkeypatch):
    class SlowStore(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            time.sleep(0.03)
            return result

    monkeypatch.setattr(operations, "BUDGET_ADMISSION_MAX_SECONDS", 0.01)
    monkeypatch.setattr(operations, "PROVIDER_BILLING_RECONCILIATION_SECONDS", 0.01)
    monkeypatch.setattr(operations, "PROVIDER_API_TIMEOUT_GRACE_SECONDS", 0.01)
    payload = {
        "id": "gen-chat-timeout",
        "model": "openai/gpt-4o-mini",
        "choices": [],
        "usage": {"cost": "0.000012"},
    }
    broker, store, transport = make_broker(
        store=SlowStore(), transport=FakeTransport([(200, payload)]),
        openrouter_shared_gate=br.OpenRouterSharedGate(1),
    )

    result = broker.execute(
        context(11), operation_id="openrouter.chat", parameters=CHAT,
        action_sequence=1, timeout_ms=50,
    )

    assert result.status == 200
    assert store.log == ["reserve", "dispatch", "settle"]
    assert 0.04 <= transport.sent[0]["timeout"] <= 0.05


def test_invalid_operation_and_action_precede_bad_timeout_conversion():
    broker, store, transport = make_broker(
        openrouter_shared_gate=br.OpenRouterSharedGate(1)
    )
    invalid_operation = broker.execute(
        context(12), operation_id="missing.operation", parameters={},
        action_sequence=0, timeout_ms="bad",
    )
    invalid_action = broker.execute(
        context(13), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=-1, timeout_ms="bad",
    )

    assert invalid_operation.call["error_code"] == "invalid_request"
    assert invalid_action.call["error_code"] == "invalid_request"
    assert store.log == [] and transport.sent == []


def test_full_185_second_budget_sets_90_second_queue_boundary(monkeypatch):
    class DeadlineGate:
        def __init__(self):
            self.deadline = None

        def acquire(self, _fingerprint, *, deadline, cancel_requested=None):
            self.deadline = deadline
            return None

    gate = DeadlineGate()
    monkeypatch.setattr(br.time, "monotonic", lambda: 100.0)
    broker, store, transport = make_broker(openrouter_shared_gate=gate)

    result = broker.execute(
        context(14), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=120_000,
    )

    assert gate.deadline == 190.0
    full_api_deadline = gate.deadline + 20.0 + 30.0 + 30.0 + 15.0
    assert full_api_deadline == 285.0
    assert result.call["reason"] == "provider_admission_deadline"
    assert store.log == [] and transport.sent == []


def test_expired_api_deadline_rejects_even_when_gate_is_idle(monkeypatch):
    clock = iter((100.0, 286.0))
    monkeypatch.setattr(br.time, "monotonic", lambda: next(clock))
    gate = br.OpenRouterSharedGate(1)
    broker, store, transport = make_broker(openrouter_shared_gate=gate)

    result = broker.execute(
        context(15), operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=1, timeout_ms=120_000,
    )

    assert result.status == 502
    assert result.call["outcome"] == "not_dispatched"
    assert result.call["reason"] == "provider_admission_deadline"
    assert store.log == [] and transport.sent == []
    assert gate._states == {}


def test_one_worker_exhausts_free_429_retries_while_other_nine_complete(
    monkeypatch, tmp_path
):
    rate_limit = {"error": {"code": "rate_limit_exceeded", "message": "limited"}}
    gate = br.OpenRouterSharedGate(
        max_concurrency=2, fallback_seconds=(0.001, 0.001)
    )
    monkeypatch.setattr(rn, "RESPONSES_RATE_LIMIT_BACKOFF_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(rn.secrets, "randbelow", lambda _bound: 0)
    brokers = []
    stores = []
    transports = []
    workers = []

    class Api:
        def __init__(self, broker, run_context):
            self.broker = broker
            self.run_context = run_context

        def provider(self, _run_id, _lease_token, frame):
            return self.broker.execute(self.run_context, **frame).to_document()

    for index in range(10):
        replies = (
            [(200, rate_limit), (200, rate_limit), (200, rate_limit)]
            if index == 0
            else [(200, response(id="gen-batch-%d" % index))]
        )
        broker, store, transport = make_broker(
            store=FakeLedgerStore(),
            transport=FakeTransport(replies),
            openrouter_shared_gate=gate,
        )
        current = lease("shared-gate-%d" % index)
        current.update(
            {
                "run_id": context(index + 20).run_id,
                "assignment_id": context(index + 20).assignment_id,
                "kind": "execute",
            }
        )
        brokers.append(broker)
        stores.append(store)
        transports.append(transport)
        workers.append(
            rn.WorkerSocketServer(
                tmp_path / ("worker-%d.sock" % index),
                Api(broker, context(index + 20)),
                rn.RunState(lease=current, lease_token="token-%d" % index),
            )
        )

    results = [None] * 10

    def run(index: int) -> None:
        results[index] = workers[index]._dispatch(
            "openrouter.responses", PARAMETERS, 120_000
        )

    threads = [threading.Thread(target=run, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()

    failed_error, failed_document = results[0]
    assert failed_error is None and failed_document["status"] == 502
    assert len(transports[0].sent) == 3
    assert [call["action_sequence"] for call in workers[0]._state.calls] == [0, 1, 2]
    assert sum(call["actual_microusd"] for call in workers[0]._state.calls) == 0
    assert all(call["outcome"] == "settled" for call in workers[0]._state.calls)

    for index in range(1, 10):
        error, document = results[index]
        assert error is None and document["status"] == 200
        assert len(transports[index].sent) == 1
        assert workers[index]._state.calls[0]["action_sequence"] == 0
        assert workers[index]._state.calls[0]["actual_microusd"] == 12
        assert stores[index].log == ["reserve", "dispatch", "settle"]


@pytest.mark.parametrize("value", ["0", "11", "x", "1.5"])
def test_shared_concurrency_environment_is_strict(monkeypatch, value):
    monkeypatch.setenv("LAB_ARENA_OPENROUTER_MAX_CONCURRENCY", value)
    with pytest.raises(ServiceError):
        _openrouter_shared_concurrency_from_environment()


def test_shared_concurrency_default_needs_no_environment_update(monkeypatch):
    monkeypatch.delenv("LAB_ARENA_OPENROUTER_MAX_CONCURRENCY", raising=False)
    assert _openrouter_shared_concurrency_from_environment() == 2
    monkeypatch.setenv("LAB_ARENA_OPENROUTER_MAX_CONCURRENCY", "10")
    assert _openrouter_shared_concurrency_from_environment() == 10
