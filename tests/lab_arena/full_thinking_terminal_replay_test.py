"""Full-thinking route evidence must not prevent exact paid-call replay."""

from copy import deepcopy
from dataclasses import replace

import pytest

from lab_arena import broker as br, reasoning_capture as rc
from tests.lab_arena import reasoning_capture_test as capture_tests
from tests.lab_arena import test_lab_arena_broker as broker_tests


def _settled_call(*, matched=False):
    context = replace(
        broker_tests.CONTEXT,
        reasoning_capture_policy=rc.POLICY,
        reasoning_route_registry=capture_tests._reviewed_registry(),
    )
    response = deepcopy(capture_tests.OK[1])
    if matched:
        response.update(
            model=broker_tests.CHAT["model"],
            provider="Example Provider",
            openrouter_metadata={"endpoints": {"available": [{
                "selected": True, "provider": "Example Provider",
                "model": broker_tests.CHAT["model"],
            }]}},
        )
    transport = broker_tests.FakeTransport([(200, response)])
    instance, store, transport = broker_tests.make_broker(
        transport=transport, judge_models=(broker_tests.CHAT["model"],)
    )
    arguments = dict(
        operation_id="openrouter.chat", parameters=broker_tests.CHAT,
        action_sequence=0, timeout_ms=30000,
    )
    first = instance.execute(context, **arguments)
    assert first.status == 200
    terminal = store.calls[first.call["call_identity"]]["terminal"]
    assert "reasoning_route_observation" in br.broker_observations(terminal)
    return instance, store, transport, context, arguments, first, terminal


@pytest.mark.parametrize("matched", [False, True])
def test_full_thinking_settlement_replays_exact_response_without_second_provider_send(matched):
    instance, store, transport, context, arguments, first, terminal = _settled_call(
        matched=matched
    )
    assert br.broker_observations(terminal)["reasoning_route_observation"]["status"] == (
        "matched" if matched else "unverified"
    )
    before = deepcopy(terminal)

    replay = instance.execute(context, **arguments)

    assert (replay.status, replay.headers, replay.body) == (
        first.status, first.headers, first.body
    )
    assert replay.call["idempotent"] is True
    assert store.calls[first.call["call_identity"]]["terminal"] == before
    assert len(transport.sent) == 1
    assert store.log.count("settle") == 1


@pytest.mark.parametrize("change", [
    lambda value: value.update(status="unknown"),
    lambda value: value.update(response_model="x" * 129),
    lambda value: value.update(requested_model=""),
    lambda value: value.update(selected_endpoint_model=123),
    lambda value: value.update(extra="unexpected"),
    lambda value: value.pop("expected_provider_name"),
])
def test_replay_rejects_malformed_reasoning_route_observation(change):
    instance, store, transport, context, arguments, first, terminal = _settled_call()
    change(br.broker_observations(terminal)["reasoning_route_observation"])

    with pytest.raises(br.BrokerError, match="broker_unavailable"):
        br._decode_terminal(terminal)
    replay = instance.execute(context, **arguments)
    assert replay.status == br.GENERIC_ERRORS["broker_unavailable"]
    assert replay.call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == 1
    assert store.log.count("settle") == 1


def test_legacy_terminal_without_observation_still_replays():
    instance, store, transport, context, arguments, first, terminal = _settled_call()
    terminal["upstream_response_receipt"]["broker_observations"].pop("reasoning_route_observation")

    replay = instance.execute(context, **arguments)

    assert (replay.status, replay.headers, replay.body) == (
        first.status, first.headers, first.body
    )
    assert len(transport.sent) == 1
    assert store.log.count("settle") == 1
