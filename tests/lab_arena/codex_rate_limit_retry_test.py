"""Bounded retries for proved free OpenRouter Responses throttles."""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import broker as br
from lab_arena import runner as rn
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker
from tests.lab_arena.test_lab_arena_runner import lease

PARAMETERS = {"model": "openai/gpt-4o-mini", "input": "research"}
BODY = base64.b64encode(b'{"error":{"code":"provider_unavailable"}}').decode()


def document(*, provider_status=429, actual=0, outcome="settled", status=502,
             error_code="provider_unavailable", funding="host", idempotent="absent",
             retry_after="absent"):
    call = {
        "operation_id": "openrouter.responses", "provider": "openrouter",
        "funding_source": funding, "outcome": outcome, "error_code": error_code,
        "provider_status": provider_status, "actual_microusd": actual,
        "status": status,
    }
    if idempotent != "absent":
        call["idempotent"] = idempotent
    if retry_after != "absent":
        call["retry_after_seconds"] = retry_after
    return {"status": status, "headers": {"content-type": "application/json"},
            "body_b64": BODY, "call": call}


class Api:
    def __init__(self, documents):
        self.documents = list(documents)
        self.frames = []
        self.called = threading.Event()

    def provider(self, _run_id, _lease_token, frame):
        self.frames.append(dict(frame))
        self.called.set()
        return self.documents.pop(0)


class ClockEvent:
    def __init__(self, clock, oversleep=0.0):
        self.clock = clock
        self.oversleep = oversleep
        self.waits = []
        self.stopped = False

    def is_set(self):
        return self.stopped

    def clear(self):
        self.stopped = False

    def set(self):
        self.stopped = True

    def wait(self, seconds):
        self.waits.append(seconds)
        self.clock[0] += seconds + self.oversleep
        return self.stopped


def server(tmp_path, api, *, kind="execute"):
    current = lease("rate-limit")
    current["kind"] = kind
    return rn.WorkerSocketServer(
        tmp_path / "worker.sock", api,
        rn.RunState(lease=current, lease_token="token"),
    )


def install_clock(monkeypatch, worker, *, oversleep=0.0):
    clock = [0.0]
    event = ClockEvent(clock, oversleep)
    worker._stopping = event
    monkeypatch.setattr(rn.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rn.secrets, "randbelow", lambda _bound: 0)
    return event


def test_worker_retries_two_free_throttles_with_new_sequences_and_shared_deadline(monkeypatch, tmp_path):
    api = Api([document(), document(), document(status=200, provider_status=200,
              actual=7, error_code=None)])
    worker = server(tmp_path, api)
    event = install_clock(monkeypatch, worker)

    error, result = worker._dispatch("openrouter.responses", PARAMETERS, 120_000)

    assert error is None and result["status"] == 200
    assert [frame["action_sequence"] for frame in api.frames] == [0, 1, 2]
    assert [frame["timeout_ms"] for frame in api.frames] == [120_000, 99_999, 59_998]
    assert event.waits == [20.001, 40.001]
    assert len(worker._state.calls) == 3


@pytest.mark.parametrize("change", [
    {"actual": 1}, {"actual": False}, {"actual": "0"},
    {"outcome": "uncertain"}, {"provider_status": 500},
    {"status": 503}, {"error_code": "call_uncertain"}, {"funding": "other"},
    {"idempotent": True}, {"idempotent": None}, {"retry_after": None},
    {"retry_after": True}, {"retry_after": -1}, {"retry_after": "30"},
    {"retry_after": 3601}, {"error_code": "miner_credentials_unavailable"},
])
def test_worker_never_retries_unproved_or_malformed_outcomes(monkeypatch, tmp_path, change):
    api = Api([document(**change)])
    worker = server(tmp_path, api)
    event = install_clock(monkeypatch, worker)

    assert worker._dispatch("openrouter.responses", PARAMETERS, 120_000)[0] is None
    assert len(api.frames) == 1 and event.waits == []


def test_valid_retry_after_is_not_shortened_and_too_long_hint_stays_terminal(monkeypatch, tmp_path):
    api = Api([document(retry_after=30), document(status=200, provider_status=200,
              actual=1, error_code=None)])
    worker = server(tmp_path, api)
    event = install_clock(monkeypatch, worker)
    worker._dispatch("openrouter.responses", PARAMETERS, 120_000)
    assert event.waits == [30.001] and len(api.frames) == 2

    api = Api([document(retry_after=100)])
    worker = server(tmp_path, api)
    event = install_clock(monkeypatch, worker)
    worker._dispatch("openrouter.responses", PARAMETERS, 120_000)
    assert event.waits == [] and len(api.frames) == 1


def test_scheduler_oversleep_recomputes_budget_before_retry(monkeypatch, tmp_path):
    api = Api([document(), document(status=200, provider_status=200, actual=1,
                                   error_code=None)])
    worker = server(tmp_path, api)
    event = install_clock(monkeypatch, worker, oversleep=75.0)
    worker._dispatch("openrouter.responses", PARAMETERS, 120_000)
    assert event.waits == [20.001] and len(api.frames) == 1


def test_inconsistent_outer_status_never_retries(monkeypatch, tmp_path):
    malformed = document()
    malformed["status"] = 200
    api = Api([malformed])
    worker = server(tmp_path, api)
    event = install_clock(monkeypatch, worker)
    worker._dispatch("openrouter.responses", PARAMETERS, 120_000)
    assert event.waits == [] and len(api.frames) == 1


def test_stop_during_backoff_prevents_a_new_provider_call(monkeypatch, tmp_path):
    api = Api([document(), document(status=200, provider_status=200, actual=1,
                                   error_code=None)])
    worker = server(tmp_path, api)
    monkeypatch.setattr(rn.secrets, "randbelow", lambda _bound: 0)
    thread = threading.Thread(
        target=worker._dispatch,
        args=("openrouter.responses", PARAMETERS, 120_000),
    )
    thread.start()
    assert api.called.wait(2)
    worker.stop()
    thread.join(2)
    assert not thread.is_alive() and len(api.frames) == 1


def test_broker_exposes_only_internal_bounded_retry_delay_after_zero_cost_settlement(monkeypatch):
    payload = {"error": {"code": "rate_limit_exceeded", "message": "limited"}}
    broker, store, _ = make_broker(
        transport=FakeTransport([(200, payload, {"retry-after": "17"})])
    )
    result = broker.execute(
        CONTEXT, operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=0, timeout_ms=120_000,
    )
    assert result.status == 502
    assert result.call["retry_after_seconds"] == 17
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert store.log == ["reserve", "dispatch", "settle"]

    broker, _store, _ = make_broker(
        transport=FakeTransport([(200, payload, {"retry-after": "invalid"})])
    )
    invalid = broker.execute(
        CONTEXT, operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=0, timeout_ms=120_000,
    )
    assert "retry_after_seconds" in invalid.call
    assert invalid.call["retry_after_seconds"] is None

    broker, _store, _ = make_broker(transport=FakeTransport([(200, payload)]))
    absent = broker.execute(
        CONTEXT, operation_id="openrouter.responses", parameters=PARAMETERS,
        action_sequence=0, timeout_ms=120_000,
    )
    assert "retry_after_seconds" not in absent.call


def test_miner_funded_proved_free_throttle_uses_the_same_retry_policy(monkeypatch, tmp_path):
    api = Api([document(funding="miner_key"), document(
        status=200, provider_status=200, actual=7, error_code=None,
        funding="miner_key",
    )])
    worker = server(tmp_path, api)
    event = install_clock(monkeypatch, worker)
    error, result = worker._dispatch("openrouter.responses", PARAMETERS, 120_000)
    assert error is None and result["status"] == 200
    assert len(api.frames) == 2 and event.waits == [20.001]


@pytest.mark.parametrize("operation,kind", [
    ("openrouter.responses", "score"), ("openrouter.chat", "execute"),
])
def test_other_operation_and_scoring_retries_remain_unchanged(monkeypatch, tmp_path, operation, kind):
    api = Api([document()])
    worker = server(tmp_path, api, kind=kind)
    event = install_clock(monkeypatch, worker)
    worker._dispatch(operation, PARAMETERS, 120_000)
    assert len(api.frames) == 1 and event.waits == []


def test_lost_api_response_does_not_retry(monkeypatch, tmp_path):
    class LostResponseApi(Api):
        def provider(self, run_id, lease_token, frame):
            self.frames.append(dict(frame))
            raise rn.RunnerError("test response loss")

    api = LostResponseApi([])
    worker = server(tmp_path, api)
    event = install_clock(monkeypatch, worker)
    assert worker._dispatch("openrouter.responses", PARAMETERS, 120_000) == ("worker_unavailable", None)
    assert len(api.frames) == 1 and event.waits == []


def test_retry_after_parser_accepts_http_date_and_rejects_invalid_hints(monkeypatch):
    monkeypatch.setattr(br.time, "time", lambda: 0.0)
    assert br._retry_after_seconds({"retry-after": "Thu, 01 Jan 1970 00:00:30 GMT"}) == 30
    assert br._retry_after_seconds({"retry-after": "3600"}) == 3600
    for value in ("3601", "999999999999", "-1", "NaN", "30, 40", " 30", "30 "):
        assert br._retry_after_seconds({"retry-after": value}) is None
