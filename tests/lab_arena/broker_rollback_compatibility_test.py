"""New paid-call records remain readable by the release being rolled back to."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import pytest

from lab_arena import broker as br
from tests.lab_arena.full_thinking_terminal_replay_test import _settled_call
from tests.lab_arena import test_lab_arena_broker as bt, reasoning_capture_test as ct


def main_decoder():
    # Keep the actual strict release decoder, not a permissive approximation.
    namespace = dict(vars(br))
    source = Path(__file__).parents[1] / "fixtures/research_review/main_203225c57_terminal_decoder.txt"
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    return namespace["_decode_terminal"]


@pytest.mark.parametrize("full_thinking", [False, True])
def test_new_real_settlement_replays_on_main_without_another_provider_charge(monkeypatch, full_thinking):
    if full_thinking:
        instance, store, transport, context, arguments, first, terminal = _settled_call(matched=True)
    else:
        instance, store, transport = bt.make_broker(transport=bt.FakeTransport([deepcopy(ct.OK)]))
        context = bt.CONTEXT
        arguments = dict(operation_id="openrouter.chat", parameters=bt.CHAT,
                         action_sequence=0, timeout_ms=30000)
        first = instance.execute(context, **arguments)
        assert first.status == 200
        terminal = store.calls[first.call["call_identity"]]["terminal"]
    capacity = store.openrouter_capacity
    # JSON bytes are the durable contract; decode after a real serialization.
    saved = json.loads(json.dumps(terminal))
    assert main_decoder()(saved) == (first.status, first.headers, first.body)
    monkeypatch.setattr(br, "_decode_terminal", main_decoder())
    replay = instance.execute(context, **arguments)
    assert (replay.status, replay.headers, replay.body) == (first.status, first.headers, first.body)
    assert replay.call["idempotent"] is True
    assert len(transport.sent) == store.log.count("dispatch") == store.log.count("settle") == 1
    assert store.openrouter_capacity == capacity


def test_old_decoder_still_rejects_unknown_response_authority():
    terminal = br._terminal_response_document(200, {}, b'{}', call_succeeded=True)
    assert main_decoder()(terminal) == (200, {}, b'{}')
    with pytest.raises(br.BrokerError, match="broker_unavailable"):
        main_decoder()(dict(terminal, unknown_response_field=True))


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(transport_elapsed_ms=True),
    lambda value: value.update(transport_elapsed_ms=-1),
    lambda value: value.update(transport_elapsed_ms=3_600_001),
    lambda value: value.update(timing_scope="all_requests"),
    lambda value: value.pop("timing_scope"),
    lambda value: value.update(status=200),
    lambda value: value.update(body_b64="e30="),
    lambda value: value.update(reasoning_route_observation={"status": "matched"}),
])
def test_nested_observations_are_closed_bounded_and_cannot_supply_response_authority(mutate):
    terminal = br._terminal_response_document(502, {}, b'{"error":"original"}', call_succeeded=False)
    observations = {"transport_elapsed_ms": 1, "timing_scope": "initial_provider_transport_send"}
    terminal["upstream_response_receipt"] = {"broker_observations": observations}
    mutate(observations)
    with pytest.raises(br.BrokerError, match="broker_unavailable"):
        br._decode_terminal(terminal)


def test_legacy_draft_observation_and_conflicts():
    terminal = br._terminal_response_document(200, {}, b'{}', call_succeeded=True)
    old = dict(terminal, transport_elapsed_ms=11, timing_scope="initial_provider_transport_send")
    assert br._decode_terminal(old) == (200, {}, b'{}')
    br._retain_broker_observations(terminal, br.broker_observations(old))
    assert br._decode_terminal(terminal) == br._decode_terminal(old)
    assert main_decoder()(terminal) == br._decode_terminal(old)
    terminal.update(transport_elapsed_ms=12, timing_scope="initial_provider_transport_send")
    with pytest.raises(br.BrokerError, match="broker_unavailable"):
        br._decode_terminal(terminal)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("full_thinking", [False, True])
def test_uncertain_transport_and_missing_cost_keep_observations_and_never_resend(failure, full_thinking):
    response = {"error": {"code": 408, "message": "request timeout"}}
    transport = bt.FakeTransport([(200, response)], fail=failure)
    instance, store, _ = bt.make_broker(transport=transport)
    arguments = dict(operation_id="openrouter.chat", parameters=bt.CHAT,
                     action_sequence=0, timeout_ms=30000)
    context = (replace(bt.CONTEXT, reasoning_capture_policy=br.reasoning_capture.POLICY,
                       reasoning_route_registry=ct._reviewed_registry()) if full_thinking else bt.CONTEXT)
    first = instance.execute(context, **arguments)
    saved = store.calls[first.call["call_identity"]]
    assert first.call["outcome"] == "uncertain"
    observations = br.broker_observations(json.loads(json.dumps(saved["uncertain_doc"])))
    if full_thinking:
        assert observations["timing_scope"] == "initial_provider_transport_send"
        assert type(observations["transport_elapsed_ms"]) is int
    else:
        assert observations == {}
    capacity = store.openrouter_capacity
    replay = instance.execute(context, **arguments)
    assert replay.call["outcome"] == "uncertain"
    assert len(transport.sent) == store.log.count("dispatch") == 1
    assert store.openrouter_capacity == capacity


@pytest.mark.parametrize("full_thinking", [False, True])
def test_completed_uncertain_deepline_response_and_later_reconciliation_replay_on_main(monkeypatch, full_thinking):
    from tests.lab_arena import deepline_score_billing_latency_test as score
    transport = score.CompletedTransport()
    instance, store, context = score.setup(transport)
    if full_thinking:
        context = replace(context, reasoning_capture_policy=br.reasoning_capture.POLICY)
    first = score.execute(instance, context)
    identity = first.call["call_identity"]
    assert first.status == 200 and first.call["outcome"] == "uncertain"
    observations = br.broker_observations(store.calls[identity]["uncertain_doc"])
    if full_thinking:
        assert observations["timing_scope"] == "initial_provider_transport_send"
    else:
        assert observations == {}
    assert main_decoder()(json.loads(json.dumps(store.responses[identity])))[2] == first.body
    monkeypatch.setattr(br, "_decode_terminal", main_decoder())
    before = len(transport.requests)
    assert score.execute(instance, context).body == first.body
    assert len(transport.requests) == before
    transport.bill_final = True
    reservation = store.calls[identity]["call_doc"]
    candidate = score._candidate(
        run_id=context.run_id, round_id=context.round_id,
        assignment_id=context.assignment_id, submission_id=context.submission_id,
        miner_hotkey=context.miner_hotkey, call_identity=identity,
        request_id=score.NATIVE, operation="parallel_search", billing_provider="parallel",
        credential_fingerprint=reservation["credential_fingerprint"],
        execution_key=reservation["deepline_execution_key"])
    assert instance.reconcile_deepline_cost(candidate) == {"status": "settled", "actual_microusd": 2000}
    replay = score.execute(instance, context)
    assert replay.body == first.body and replay.call["actual_microusd"] == 2000
    assert [request["method"] for request in transport.requests] == ["POST", "GET", "GET"]
    assert store.log.count("dispatch") == store.log.count("reconcile") == 1


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("full_thinking", [False, True])
def test_failed_and_compact_known_charge_terminal_replays_on_main(monkeypatch, compact, full_thinking):
    class EdgeStore(bt.FakeLedgerStore):
        def settle_call(self, **kwargs):
            if compact and kwargs["terminal_response"]["status"] == 200:
                raise br.ArenaStoreError("rpc lab_arena_settle_call failed: HTTP 403")
            return super().settle_call(**kwargs)

    if compact:
        responses = [(200, bt._LARGE_SCRAPINGDOG_PAGE)]
        arguments = dict(operation_id="scrapingdog.scrape", parameters={"url": "https://example.com/about"},
                         action_sequence=0, timeout_ms=5000)
    else:
        responses = [(200, {"error": {"code": 400, "message": "request rejected"},
                            "usage": {"cost": "0.0000091"}})]
        arguments = dict(operation_id="openrouter.chat", parameters=bt.CHAT,
                         action_sequence=0, timeout_ms=30000)
    instance, store, transport = bt.make_broker(store=EdgeStore(), transport=bt.FakeTransport(responses))
    context = (replace(bt.CONTEXT, reasoning_capture_policy=br.reasoning_capture.POLICY,
                       reasoning_route_registry=ct._reviewed_registry()) if full_thinking else bt.CONTEXT)
    first = instance.execute(context, **arguments)
    assert first.call["outcome"] == "settled" and first.status in (400, 502)
    terminal = store.calls[first.call["call_identity"]]["terminal"]
    assert main_decoder()(json.loads(json.dumps(terminal)))[2] == first.body
    observations = br.broker_observations(terminal)
    if full_thinking:
        assert observations["timing_scope"] == "initial_provider_transport_send"
    else:
        assert observations == {}
    monkeypatch.setattr(br, "_decode_terminal", main_decoder())
    replay = instance.execute(context, **arguments)
    assert replay.body == first.body and replay.status == first.status
    assert replay.call["actual_microusd"] == first.call["actual_microusd"]
    assert len(transport.sent) == store.log.count("dispatch") == 1


def test_legacy_draft_full_thinking_terminal_replays_without_a_second_charge():
    instance, store, transport, context, arguments, first, terminal = _settled_call(matched=True)
    observations = terminal.pop("upstream_response_receipt")["broker_observations"]
    terminal.update(observations)
    assert br._decode_terminal(terminal) == (first.status, first.headers, first.body)
    replay = instance.execute(context, **arguments)
    assert replay.body == first.body and replay.call["idempotent"] is True
    assert len(transport.sent) == store.log.count("settle") == 1


def test_optional_upstream_capture_and_oversize_fallback_preserve_observations():
    terminal = br._terminal_response_document(502, {}, b'{}', call_succeeded=False)
    observations = {"transport_elapsed_ms": 41, "timing_scope": "initial_provider_transport_send"}
    br._retain_broker_observations(terminal, observations)
    capture = {"status": 503, "body_b64": "e30=", "capture_kind": "upstream_response"}
    br._retain_upstream_response_receipt(terminal, capture)
    assert br.broker_observations(terminal) == observations
    assert terminal["upstream_response_receipt"]["body_b64"] == "e30="
    assert main_decoder()(terminal)[2] == b'{}'
    br._retain_upstream_response_receipt(terminal, dict(capture, body_b64="x" * (4 * 1024 * 1024)))
    assert br.broker_observations(terminal) == observations
    assert terminal["upstream_response_receipt"]["status"] == "unavailable"
    assert terminal["upstream_response_receipt"]["reason"] == "response_size_limit"
    assert main_decoder()(terminal)[2] == b'{}'
