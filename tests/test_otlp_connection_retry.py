"""Bound OTLP-only disconnect recovery without hiding exhausted failures."""

import logging
import random
from types import SimpleNamespace

import pytest
import requests
from opentelemetry.exporter.otlp.proto.http import trace_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult

from gateway.observability import otel_bootstrap as bootstrap


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []
        self.overshoot = 0

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay + self.overshoot


def response(status):
    result = requests.Response()
    result.status_code = status
    result._content = b"synthetic response"
    return result


@pytest.fixture
def setup_exporter(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(bootstrap, "time", SimpleNamespace(
        monotonic=clock.monotonic, sleep=clock.sleep,
    ))
    def jitter(low, high):
        assert (low, high) == (0.1, 0.2)
        return 0.15
    monkeypatch.setattr(random, "uniform", jitter)
    monkeypatch.setenv("GATEWAY_OTEL_ENDPOINT", "https://example.invalid/v1/traces")
    monkeypatch.setenv("GATEWAY_OTEL_TOKEN", "synthetic-test-token")
    calls = []
    outcomes = []
    def post(session, *args, **kwargs):
        calls.append((session, args, kwargs))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            clock.now += 2
            raise outcome
        return outcome
    monkeypatch.setattr(requests.Session, "post", post)
    exporter = bootstrap._build_otlp_exporter(bootstrap._GATEWAY_LOG)
    yield exporter, clock, calls, outcomes
    exporter.shutdown()


def test_disconnect_retries_identical_bytes_with_remaining_timeout(setup_exporter):
    exporter, clock, calls, outcomes = setup_exporter
    outcomes.extend([requests.ConnectionError("synthetic disconnect"), response(200)])
    payload = b"synthetic immutable protobuf bytes"
    assert exporter._export_serialized_spans(payload) is SpanExportResult.SUCCESS
    assert len(calls) == 2
    assert calls[0][2]["data"] is calls[1][2]["data"] is payload
    assert calls[0][2]["timeout"] == exporter._timeout
    assert calls[1][2]["timeout"] == pytest.approx(exporter._timeout - 2.15)
    assert clock.sleeps == [0.15]
    assert calls[0][0] is calls[1][0]
    assert calls[0][0].headers["Authorization"] == "Bearer synthetic-test-token"
    assert calls[0][0].trust_env is True
    assert calls[0][0].proxies == {}
    assert calls[0][1] == calls[1][1]
    assert {k: v for k, v in calls[0][2].items() if k != "timeout"} == {
        k: v for k, v in calls[1][2].items() if k != "timeout"
    }


@pytest.mark.parametrize("failure", [
    requests.Timeout("synthetic timeout"), requests.ConnectTimeout("synthetic connect timeout"),
    requests.ReadTimeout("synthetic read timeout"),
])
def test_timeouts_do_not_get_an_additional_transport_attempt(setup_exporter, failure):
    exporter, clock, calls, outcomes = setup_exporter
    outcomes.append(failure)
    with pytest.raises(type(failure)) as caught:
        exporter._export_serialized_spans(b"synthetic payload")
    assert caught.value is failure
    assert len(calls) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("overshoot", [False, True])
def test_exhausted_original_deadline_prevents_second_attempt(setup_exporter, monkeypatch, overshoot):
    exporter, clock, calls, outcomes = setup_exporter
    failure = requests.ConnectionError("synthetic disconnect")
    outcomes.append(failure)
    if overshoot:
        clock.overshoot = exporter._timeout
    else:
        # The first request consumed all but less than the retry backoff.
        real_post = requests.Session.post
        def post(*args, **kwargs):
            clock.now += exporter._timeout - 2.1
            return real_post(*args, **kwargs)
        monkeypatch.setattr(requests.Session, "post", post)
    with pytest.raises(requests.ConnectionError) as caught:
        exporter._export_serialized_spans(b"synthetic payload")
    assert caught.value is failure
    assert len(calls) == 1
    assert clock.sleeps == ([0.15] if overshoot else [])


@pytest.mark.parametrize("status", [400, 401, 403])
def test_permanent_http_responses_are_not_retried(setup_exporter, status, caplog):
    exporter, clock, calls, outcomes = setup_exporter
    outcomes.append(response(status))
    with caplog.at_level(logging.ERROR):
        assert exporter._export_serialized_spans(b"synthetic payload") is SpanExportResult.FAILURE
    assert len(calls) == 1
    assert clock.sleeps == []
    assert "Failed to export batch code" in caplog.text


def test_http_status_backoff_remains_owned_by_sdk(setup_exporter, monkeypatch):
    exporter, clock, calls, outcomes = setup_exporter
    sdk_sleeps = []
    monkeypatch.setattr(trace_exporter, "sleep", sdk_sleeps.append)
    outcomes.extend([response(503), response(200)])
    assert exporter._export_serialized_spans(b"synthetic payload") is SpanExportResult.SUCCESS
    assert len(calls) == 2
    assert len(sdk_sleeps) == 1
    assert clock.sleeps == []


def test_exhausted_failure_is_logged_and_queue_worker_continues(setup_exporter, caplog):
    exporter, _clock, calls, outcomes = setup_exporter
    outcomes.extend([
        requests.ConnectionError("first synthetic disconnect"),
        requests.ConnectionError("second synthetic disconnect"), response(200),
    ])
    processor = bootstrap._build_span_processor(exporter, simple=False)
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("synthetic.otlp.retry")
    try:
        with caplog.at_level(logging.ERROR):
            tracer.start_span("first synthetic span").end()
            assert provider.force_flush(timeout_millis=1000)
        assert len(calls) == 2
        assert "Exception while exporting Span batch." in caplog.text
        failures = [record for record in caplog.records if record.exc_info]
        assert failures[-1].exc_info[0] is requests.ConnectionError
        assert len(processor.queue) == 0
        tracer.start_span("second synthetic span").end()
        assert provider.force_flush(timeout_millis=1000)
        assert len(calls) == 3
        assert len(processor.queue) == 0
    finally:
        provider.shutdown()


@pytest.mark.parametrize("timeout,data", [(None, b"bytes"), ((1, 2), b"bytes"), (10, bytearray(b"mutable"))])
def test_unsupported_request_shapes_are_not_replayed(setup_exporter, timeout, data):
    exporter, clock, calls, outcomes = setup_exporter
    outcomes.append(requests.ConnectionError("synthetic disconnect"))
    with pytest.raises(requests.ConnectionError):
        exporter._session.post("https://example.invalid/v1/traces", data=data, timeout=timeout)
    assert len(calls) == 1
    assert clock.sleeps == []
