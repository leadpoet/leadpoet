"""Bounded retries for proved free OpenRouter Responses throttles."""

from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import broker as br
from lab_arena import contracts, runtime, shim
from lab_arena import runner as rn
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT,
    LUNA_RESPONSES,
    FakeTransport,
    luna_price_table,
    make_broker,
)
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime,
    FakeApi,
    lease,
    make_config,
)

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


def identified_document(sequence, **changes):
    result = document(**changes)
    result["call"]["call_identity"] = contracts.document_hash(
        ["responses-retry", sequence]
    )
    result["call"]["action_sequence"] = sequence
    if result["status"] == 200:
        result["body_b64"] = base64.b64encode(
            b'{"id":"resp-recovered","status":"completed","output":[]}'
        ).decode()
    return result


class ResponsesThenTerminalRuntime(BridgingRuntime):
    def __init__(self, statuses, *, timed_out=False):
        super().__init__(output=None, exit_code=1, timed_out=timed_out, calls=0)
        self.statuses = tuple(statuses)

    def run_icp(self, spec, **_kwargs):
        self.specs.append(spec)
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            for expected in self.statuses:
                status, _headers, _body = shim.dispatch(
                    "openrouter.responses", PARAMETERS, 120_000
                )
                assert status == expected
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        return runtime.fake_result(
            exit_code=self.exit_code,
            timed_out=self.timed_out,
            output_bytes=None,
            stderr=b"model stopped without output",
        )


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


@pytest.mark.parametrize(
    "timed_out,expected_terminal",
    [(False, "model_error"), (True, "model_timeout")],
)
def test_recovered_transparent_retry_does_not_replace_later_model_terminal(
    monkeypatch, tmp_path, timed_out, expected_terminal
):
    documents = [
        identified_document(0, retry_after=0),
        identified_document(
            1,
            provider_status=200,
            actual=7,
            status=200,
            error_code=None,
            idempotent=False,
        ),
    ]
    statuses = [200]
    if not timed_out:
        documents.append(
            identified_document(
                2,
                provider_status=None,
                actual=0,
                outcome="refused",
                status=402,
                error_code="budget_refused",
                idempotent=False,
            )
        )
        statuses.append(402)
    api = FakeApi([lease("recovered-model-terminal")], broker_documents=documents)
    (tmp_path / "work").mkdir()
    monkeypatch.setattr(rn.secrets, "randbelow", lambda _bound: 0)

    assert rn.Runner(
        make_config(
            tmp_path,
            api,
            ResponsesThenTerminalRuntime(statuses, timed_out=timed_out),
        )
    ).run_once() == 1

    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == expected_terminal
    assert result["resource_summary"]["provider_call_count"] == len(documents)
    assert len(api.provider_frames) == len(documents)


@pytest.mark.parametrize(
    "documents,statuses",
    [
        (
            [
                identified_document(0, retry_after=0),
                identified_document(
                    1,
                    provider_status=200,
                    actual=7,
                    status=200,
                    error_code=None,
                    idempotent=False,
                ),
                identified_document(
                    2,
                    provider_status=502,
                    actual=0,
                    status=502,
                    error_code="provider_unavailable",
                    idempotent=False,
                ),
            ],
            [200, 502],
        ),
        (
            [
                document(retry_after=0),
                identified_document(
                    1,
                    provider_status=200,
                    actual=7,
                    status=200,
                    error_code=None,
                    idempotent=False,
                ),
                identified_document(
                    2,
                    provider_status=None,
                    actual=0,
                    outcome="refused",
                    status=402,
                    error_code="budget_refused",
                    idempotent=False,
                ),
            ],
            [200, 402],
        ),
        (
            [
                identified_document(0, retry_after=0),
                identified_document(1, retry_after=0),
                identified_document(2, retry_after=0),
            ],
            [502],
        ),
        (
            [
                identified_document(
                    0,
                    provider_status=None,
                    actual=0,
                    status=502,
                    error_code="provider_unavailable",
                    idempotent=False,
                ),
                identified_document(
                    1,
                    provider_status=None,
                    actual=0,
                    outcome="refused",
                    status=402,
                    error_code="budget_refused",
                    idempotent=False,
                ),
            ],
            [502, 402],
        ),
    ],
    ids=(
        "later-visible-infrastructure",
        "missing-retry-identity",
        "exhausted-final-rate-limit",
        "unproved-zero-cost-502-before-quota",
    ),
)
def test_only_proved_recovered_retry_is_excluded_from_infrastructure_failure(
    monkeypatch, tmp_path, documents, statuses
):
    api = FakeApi([lease("unrecovered-provider")], broker_documents=documents)
    (tmp_path / "work").mkdir()
    monkeypatch.setattr(rn.secrets, "randbelow", lambda _bound: 0)

    assert rn.Runner(
        make_config(
            tmp_path,
            api,
            ResponsesThenTerminalRuntime(statuses),
        )
    ).run_once() == 1

    result = api.completions[0]["body"]["result"]
    assert result["terminal_status"] == "provider_error"
    assert result["resource_summary"]["provider_call_count"] == len(documents)
    assert len(api.provider_frames) == len(documents)


def test_responses_timeout_allows_existing_second_free_throttle_recovery(monkeypatch, tmp_path):
    class TimedApi(Api):
        def __init__(self, documents, clock, durations):
            super().__init__(documents)
            self.clock = clock
            self.durations = iter(durations)

        def provider(self, run_id, lease_token, frame):
            self.clock[0] += next(self.durations)
            return super().provider(run_id, lease_token, frame)

    def dispatch(timeout_ms):
        clock = [0.0]
        api = TimedApi([
            document(),
            document(),
            document(status=200, provider_status=200, actual=7, error_code=None),
        ], clock, [40.0, 10.0, 1.0])
        worker = server(tmp_path, api)
        event = ClockEvent(clock)
        worker._stopping = event
        monkeypatch.setattr(rn.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(rn.secrets, "randbelow", lambda _bound: 0)
        outcome = worker._dispatch("openrouter.responses", PARAMETERS, timeout_ms)
        return outcome, api, event, clock[0]

    old_outcome, old_api, old_event, old_elapsed = dispatch(120_000)
    assert old_outcome[1]["status"] == 502
    assert len(old_api.frames) == 2
    assert old_event.waits == [20.001]
    assert old_elapsed == pytest.approx(70.001)

    new_outcome, new_api, new_event, new_elapsed = dispatch(300_000)
    assert new_outcome[0] is None and new_outcome[1]["status"] == 200
    assert [frame["action_sequence"] for frame in new_api.frames] == [0, 1, 2]
    assert new_event.waits == [20.001, 40.001]
    assert new_elapsed == pytest.approx(111.002)


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


def test_worker_retries_luna_throttle_with_same_bounded_host_policy(
    monkeypatch, tmp_path
):
    success = {
        "id": "gen-luna-after-throttle",
        "object": "response",
        "created_at": 1789488000,
        "status": "completed",
        "model": br.OPENROUTER_LUNA_RESPONSES_MODEL,
        "error": None,
        "output": [
            {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "ok",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
            "cost": "0.00025",
        },
    }
    throttle = {"error": {"code": "rate_limit_exceeded", "message": "limited"}}
    broker, store, transport = make_broker(
        transport=FakeTransport([(200, throttle), (200, success)])
    )
    broker._price_table = luna_price_table()

    class BrokerApi:
        def __init__(self):
            self.frames = []

        def provider(self, _run_id, _lease_token, frame):
            self.frames.append(dict(frame))
            return broker.execute(CONTEXT, **frame).to_document()

    api = BrokerApi()
    worker = server(tmp_path, api)
    event = install_clock(monkeypatch, worker)

    error, result = worker._dispatch(
        "openrouter.responses", LUNA_RESPONSES, 300_000
    )

    assert error is None and result["status"] == 200
    assert [frame["action_sequence"] for frame in api.frames] == [0, 1]
    assert event.waits == [20.001]
    assert len(store.calls) == 2
    assert [call["kind"] for call in store.calls.values()] == [
        "settlement",
        "settlement",
    ]
    assert [call["actual"] for call in store.calls.values()] == [0, 250]
    assert len(transport.sent) == 2
    for request in transport.sent:
        body = json.loads(request["body"])
        assert body["provider"]["only"] == ["azure/us", "azure/eu"]
        assert "order" not in body["provider"]
        assert "require_parameters" not in body["provider"]
        assert body["provider"]["max_price"] == {
            "prompt": 0.275,
            "completion": 1.32,
            "request": 0,
        }
        assert body["provider"]["zdr"] is True
        assert body["provider"]["data_collection"] == "deny"


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
