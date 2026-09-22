"""Long native calls keep caller limits, quota identity and uncertain billing."""
from types import SimpleNamespace

import pytest

from lab_arena import broker as br, operations, runner, shim
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker


@pytest.mark.parametrize("requested_ms,expected_seconds", [
    (20_000, 20.0), (240_000, 240.0), (780_000, 240.0),
])
def test_native_timeout_keeps_unknown_charge_and_does_not_repeat_paid_call(
    monkeypatch, requested_ms, expected_seconds,
):
    elapsed = [0.0]
    monkeypatch.setattr(br, "time", SimpleNamespace(
        time=lambda: elapsed[0], monotonic=lambda: elapsed[0],
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
    ))
    transport = FakeTransport(fail=True)
    broker, store, _ = make_broker(transport=transport)
    parameters = {"tool": "exa_search", "payload": {"query": "example"}}
    frame = shim.build_operation_frame("deepline.execute", parameters, requested_ms)
    operation, normalized, bounded_ms = shim.decode_operation_frame(frame)
    assert bounded_ms == expected_seconds * 1000
    result = broker.execute(CONTEXT, operation_id=operation, parameters=normalized,
                            action_sequence=0, timeout_ms=bounded_ms)
    paid = [item for item in transport.sent if item["method"] == "POST"]
    assert len(paid) == 1 and paid[0]["timeout"] == expected_seconds
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert store.openrouter_capacity == 0  # No unsupported price or zero charge.
    assert len(store.calls) == 1
    record = next(iter(store.calls.values()))
    assert record["kind"] == "uncertain" and record["amount"] == 10_000_000
    replay = broker.execute(CONTEXT, operation_id=operation, parameters=normalized,
                            action_sequence=0, timeout_ms=bounded_ms)
    assert replay.status == 409 and replay.call["error_code"] == "call_uncertain"
    assert len([item for item in transport.sent if item["method"] == "POST"]) == 1
    assert len(store.calls) == 1


@pytest.mark.parametrize("requested_ms,expected_read_seconds", [
    (20_000, 85.0), (240_000, 305.0), (780_000, 305.0),
])
def test_api_envelope_covers_only_the_bounded_native_request(
    requested_ms, expected_read_seconds,
):
    observed = []
    class Client:
        def post(self, _url, **kwargs):
            observed.append(kwargs["timeout"].read)
            return SimpleNamespace(status_code=200, json=lambda: {"status": "ok"})
    api = runner.HttpArenaApiClient("http://localhost", client=Client())
    api.provider("run-1", "a" * 64, {
        "operation_id": "deepline.execute", "parameters": {},
        "timeout_ms": requested_ms, "action_sequence": 0,
    })
    assert observed == [expected_read_seconds]
    assert operations.OPERATIONS["openrouter.responses"].timeout_seconds == 600
    assert runner.MAX_PROVIDER_API_TIMEOUT_SECONDS == 665
    assert operations.OPERATIONS["scrapingdog.scrape"].timeout_seconds == 60
