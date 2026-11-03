"""Retry a proved reservation rollback without replaying a paid provider call."""

import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import broker as br, scoring, store as store_module
from lab_arena.runner import HttpArenaApiClient
from lab_arena.store import (
    ArenaStore, ArenaStoreError, ArenaStoreUnavailable, DEADLOCK_RETRIES,
    PostgrestTransport,
)
from tests.lab_arena.test_lab_arena_broker import (
    CHAT, CONTEXT, HOST_KEYS, FakeLedgerStore, FakeTransport, price_table,
)

RESERVATIONS = ("lab_arena_reserve_call", "lab_arena_reserve_judgment_call")
PRE_DISPATCH = RESERVATIONS + ("lab_arena_mark_dispatched",)
LOCK_TIMEOUT = {"code": "55P03", "message": "canceling statement due to lock timeout"}


@pytest.mark.parametrize("function", PRE_DISPATCH)
@pytest.mark.parametrize("code", ["55P03", "40P01"])
@pytest.mark.parametrize("failures", [1, DEADLOCK_RETRIES])
def test_definite_rollback_repeats_identical_rpc_with_existing_bound(function, code, failures):
    requests = []
    params = {"p_call_identity": "sha256:" + "a" * 64,
              "p_lease_token_hash": "sha256:" + "b" * 64}

    def handle(request):
        requests.append(request)
        return httpx.Response(500, json={**LOCK_TIMEOUT, "code": code}) if len(requests) <= failures else httpx.Response(200, json={"status": "reserved"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        transport = PostgrestTransport("https://db.example", service_key="sb_secret_test", http_client=client)
        assert transport.rpc(function, params) == {"status": "reserved"}
        assert transport.deadlock_retries == failures
    assert len(requests) == failures + 1
    assert {request.content for request in requests} == {requests[0].content}
    assert json.loads(requests[0].content) == params
    assert {request.url.path for request in requests} == {"/rest/v1/rpc/" + function}


@pytest.mark.parametrize("function", PRE_DISPATCH)
def test_persistent_lock_timeout_stops_at_existing_retry_bound(function):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(500, json=LOCK_TIMEOUT))) as client:
        transport = PostgrestTransport("https://db.example", service_key="sb_secret_test", http_client=client)
        with pytest.raises(ArenaStoreUnavailable, match="55P03") as failed:
            transport.rpc(function, {})
        assert failed.value.transaction_rolled_back is True
        assert transport.deadlock_retries == DEADLOCK_RETRIES
    assert len(requests) == DEADLOCK_RETRIES + 1


@pytest.mark.parametrize("function,status,body", [
    *[(function, 500, LOCK_TIMEOUT) for function in (
        "lab_arena_whoami", "lab_arena_settle_call", "lab_arena_open_scoring",
    )],
    *[(function, status, body) for function in PRE_DISPATCH for status, body in (
        (401, LOCK_TIMEOUT), (403, LOCK_TIMEOUT), (503, LOCK_TIMEOUT),
        (500, {"code": "42501", "message": "permission denied"}),
        (500, {"code": "P0001", "message": "unknown failure"}),
        (500, {"message": "canceling statement due to lock timeout"}),
    )],
])
def test_other_rpc_auth_unknown_or_untyped_failure_never_retries(function, status, body):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(status, json=body))) as client:
        transport = PostgrestTransport("https://db.example", service_key="sb_secret_test", http_client=client)
        with pytest.raises(ArenaStoreError):
            transport.rpc(function, {})
        assert transport.deadlock_retries == 0
    assert len(requests) == 1


@pytest.mark.parametrize("function", PRE_DISPATCH)
def test_ambiguous_transport_failure_remains_for_existing_caller_readback(function):
    requests = []

    def handle(request):
        requests.append(request)
        raise httpx.ReadTimeout("response may have committed")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        transport = PostgrestTransport("https://db.example", service_key="sb_secret_test", http_client=client)
        with pytest.raises(ArenaStoreUnavailable):
            transport.rpc(function, {})
        assert transport.deadlock_retries == 0
    assert len(requests) == 1


@pytest.mark.parametrize("judgment", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("dispatch_marker", [False, True])
def test_transport_to_broker_reserves_dispatches_and_charges_once(monkeypatch, judgment, persistent, dispatch_marker):
    rpc_requests = []
    function = "lab_arena_mark_dispatched" if dispatch_marker else RESERVATIONS[int(judgment)]
    ledger = FakeLedgerStore()
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    clock.sleep = lambda seconds: setattr(clock, "now", clock.now + seconds)
    monkeypatch.setattr(store_module, "time", clock)
    monkeypatch.setattr(br, "time", clock)

    def handle(request):
        rpc_requests.append(request)
        assert request.url.path == "/rest/v1/rpc/" + function
        if len(rpc_requests) <= DEADLOCK_RETRIES or persistent:
            assert (ledger.calls == {}) is (not dispatch_marker)
            clock.now += 5.0  # Production lab_arena_service lock_timeout.
            return httpx.Response(500, json=LOCK_TIMEOUT)
        params = json.loads(request.content)
        write = FakeLedgerStore.mark_dispatched if dispatch_marker else FakeLedgerStore.reserve_call
        result = write(ledger, **{
            key.removeprefix("p_"): value for key, value in params.items()
        })
        return httpx.Response(200, json=result)

    class Store:
        def __init__(self, inner):
            self.inner = inner

        def reserve_call(self, **kwargs):
            assert not judgment
            return ledger.reserve_call(**kwargs) if dispatch_marker else self.inner.reserve_call(**kwargs)

        def reserve_judgment_call(self, **kwargs):
            assert judgment
            return ledger.reserve_call(**kwargs) if dispatch_marker else self.inner.reserve_judgment_call(**kwargs)

        def mark_dispatched(self, **kwargs):
            return self.inner.mark_dispatched(**kwargs) if dispatch_marker else ledger.mark_dispatched(**kwargs)

        def __getattr__(self, name):
            return getattr(ledger, name)

    policy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2")
    model = next(iter(policy["judge_models"].values()))
    table = price_table()
    table["models"][model] = dict(table["models"][CHAT["model"]])
    # Native-search judges need their fee before this fixture can reach the
    # reservation/dispatch RPC boundary being tested.
    if model in {"perplexity/sonar", "perplexity/sonar-pro"}:
        table["models"][model]["web_search"] = "0.005"
    context = replace(CONTEXT, kind="score" if judgment else "execute", round_id="round",
        judgment_cache_scope={
            "round_id": "round", "evaluation_date": "2026-10-12",
            "scorer_image_digest": "sha256:" + "c" * 64,
            "scorer_image_reference": "registry/scorer@sha256:" + "c" * 64,
            "scorer_policy": policy,
        } if judgment else None)
    provider = FakeTransport([(200, {"id": "gen", "model": model, "choices": [],
        "usage": {"prompt_tokens": 20, "completion_tokens": 50, "cost": "0.000033"}})])
    provider_request = provider.send

    def full_provider_timeout(*args, **kwargs):
        clock.now += 30.0
        return provider_request(*args, **kwargs)

    monkeypatch.setattr(provider, "send", full_provider_timeout)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        database = PostgrestTransport("https://db.example", service_key="sb_secret_test", http_client=client)
        broker = br.Broker(store=Store(ArenaStore(database)), key_for=lambda provider: HOST_KEYS[provider],
            price_table=table, judge_models=[model], transport=provider)
        if persistent:
            with pytest.raises(ArenaStoreUnavailable, match="55P03") as failed:
                broker.execute(context, operation_id="openrouter.chat", parameters={**CHAT, "model": model}, action_sequence=36, timeout_ms=30_000)
            assert failed.value.transaction_rolled_back is True
            assert provider.sent == [] and ledger.log.count("settle") == 0
            if dispatch_marker:
                assert len(ledger.calls) == 1
                assert next(iter(ledger.calls.values()))["kind"] == "reservation"
                assert ledger.log == ["reserve"]
            else:
                assert ledger.calls == {}
            assert clock.now == pytest.approx(20.06)
        else:
            result = broker.execute(context, operation_id="openrouter.chat", parameters={**CHAT, "model": model}, action_sequence=36, timeout_ms=30_000)
            assert result.status == 200
            assert ledger.log == ["reserve", "dispatch", "settle"]
            assert len(provider.sent) == len(ledger.calls) == 1
            assert ledger.calls[result.call["call_identity"]]["actual"] == result.call["actual_microusd"] == 33
            # Exercise the real runner timeout calculation after three 5s
            # rollbacks plus a full admitted 30s paid provider request.
            api_timeouts = []
            with httpx.Client(transport=httpx.MockTransport(lambda request:
                api_timeouts.append(request.extensions["timeout"]["read"])
                or httpx.Response(200, json=result.to_document()))) as api_http:
                api = HttpArenaApiClient("http://localhost", client=api_http)
                api.provider(context.run_id, "lease", {
                    "operation_id": "openrouter.chat", "parameters": {**CHAT, "model": model},
                    "action_sequence": 36, "timeout_ms": 30_000,
                })
            assert clock.now == pytest.approx(45.06)
            assert api_timeouts == [95.0]
            assert clock.now < api_timeouts[0]
        assert database.deadlock_retries == DEADLOCK_RETRIES
    assert len(rpc_requests) == DEADLOCK_RETRIES + 1
    assert len({request.content for request in rpc_requests}) == 1


@pytest.mark.parametrize("failure", ["transport_before", "transport_after", "non_json_after", "http503_after", "idempotent"])
def test_broker_never_replays_ambiguous_or_committed_dispatch_marker(failure):
    requests = []
    ledger = FakeLedgerStore()
    provider = FakeTransport()

    def handle(request):
        requests.append(request)
        assert request.url.path == "/rest/v1/rpc/lab_arena_mark_dispatched"
        if failure == "transport_before":
            raise httpx.ReadTimeout("unproved marker outcome")
        params = {key.removeprefix("p_"): value for key, value in json.loads(request.content).items()}
        ledger.mark_dispatched(**params)
        if failure == "transport_after":
            raise httpx.ReadTimeout("committed marker response lost")
        if failure == "non_json_after":
            return httpx.Response(200, content=b"not-json")
        if failure == "http503_after":
            return httpx.Response(503, json=LOCK_TIMEOUT)
        return httpx.Response(200, json=ledger.mark_dispatched(**params))

    class Store:
        def __init__(self, database):
            self.database = database

        def mark_dispatched(self, **kwargs):
            return self.database.mark_dispatched(**kwargs)

        def __getattr__(self, name):
            return getattr(ledger, name)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        database = PostgrestTransport("https://db.example", service_key="sb_secret_test", http_client=client)
        broker = br.Broker(store=Store(ArenaStore(database)),
            key_for=lambda provider: HOST_KEYS[provider], price_table=price_table(), transport=provider)
        arguments = dict(operation_id="openrouter.chat", parameters=CHAT, action_sequence=36, timeout_ms=30_000)
        if failure == "idempotent":
            result = broker.execute(CONTEXT, **arguments)
            assert result.status == 409 and result.call["outcome"] == "uncertain"
        else:
            with pytest.raises(ArenaStoreError):
                broker.execute(CONTEXT, **arguments)
        assert database.deadlock_retries == 0
    assert len(requests) == 1
    assert provider.sent == [] and ledger.log.count("settle") == 0
    assert len(ledger.calls) == 1
    assert next(iter(ledger.calls.values()))["kind"] == (
        "reservation" if failure == "transport_before" else "dispatch")
