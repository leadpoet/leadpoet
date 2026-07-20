"""Tests for research_lab/openrouter_telemetry.py (trajectoryimprovements.md P1/P8/P19)."""

from __future__ import annotations

import io
import json
import sys
import types
from urllib.error import HTTPError, URLError

import pytest

from research_lab import openrouter_telemetry as ot


TRACE_PREFIX = "s3://telemetry-bucket/lab"


@pytest.fixture
def fake_boto3(monkeypatch):
    puts: list[dict] = []

    class _Client:
        def put_object(self, **kwargs):
            puts.append(kwargs)

    fake = types.ModuleType("boto3")
    fake.client = lambda name, **kwargs: _Client()
    monkeypatch.setitem(sys.modules, "boto3", fake)
    # fresh uploader per test so inert/disabled state never leaks
    monkeypatch.setattr(ot, "_uploader", ot._TelemetryTraceUploader())
    return puts


@pytest.fixture
def trace_env(monkeypatch):
    monkeypatch.setenv(ot.RAW_TRACE_S3_PREFIX_ENV, TRACE_PREFIX)
    monkeypatch.setenv(ot.RAW_TRACE_KMS_KEY_ENV, "kms-key-9")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(responses):
    def open_fn(req, timeout=0):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)

    return open_fn


def _chat_response(content='{"ok": true}', reasoning=None, cost=0.002):
    message = {"content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": "gen-1",
        "model": "test/model",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19, "cost": cost},
    }


def _call(**overrides):
    kwargs = dict(
        api_key="sk-or-test-1",
        model_id="test/model",
        messages=[{"role": "user", "content": "hello"}],
        channel="qualification",
        purpose="test_purpose",
        stage="scorer_judgment",
        timeout_seconds=5,
        max_tokens=100,
    )
    kwargs.update(overrides)
    return ot.call_openrouter_chat(**kwargs)


def test_success_captures_trace_and_provider_usage(fake_boto3, trace_env):
    result = _call(opener=_opener([_chat_response(reasoning="I thought about it")]))
    ot.flush_telemetry_traces()

    assert result.content == '{"ok": true}'
    assert result.reasoning_returned is True
    assert result.cost_microusd == 2000
    usage = result.provider_usage
    assert usage["channel"] == "qualification"
    assert usage["purpose"] == "test_purpose"
    assert usage["teacher_model_flag"] is True  # scorer_judgment stage
    assert usage["call_emitter"] == "code"
    assert usage["total_tokens"] == 19
    capture = usage["reasoning_capture"]
    assert capture["requested"] is True and capture["returned"] is True
    assert capture["reasoning_hashes"][0].startswith("sha256:")
    assert usage["raw_trace_ref"]["s3_ref"].startswith(
        "s3://telemetry-bucket/lab/telemetry/qualification/"
    )
    assert len(fake_boto3) == 1
    put = fake_boto3[0]
    assert put["ServerSideEncryption"] == "aws:kms"
    assert put["SSEKMSKeyId"] == "kms-key-9"
    payload = json.loads(put["Body"])
    assert payload["call_emitter"] == "code"
    assert payload["request"]["body"]["include_reasoning"] is True
    # secrets never reach the trace store
    assert "sk-or-test-1" not in put["Body"].decode("utf-8")


def test_reasoning_unsupported_retries_and_records_drop(fake_boto3, trace_env):
    error = HTTPError(
        ot.OPENROUTER_CHAT_COMPLETIONS_URL,
        400,
        "Bad Request",
        {},
        io.BytesIO(b'{"error": "include_reasoning is not supported"}'),
    )
    result = _call(opener=_opener([error, _chat_response()]))
    ot.flush_telemetry_traces()

    assert result.reasoning_request_dropped is True
    capture = result.provider_usage["reasoning_capture"]
    assert capture["request_dropped"] is True
    assert capture["drop_error_hash"].startswith("sha256:")
    # both attempts captured: the http_error and the retry response
    outcomes = {json.loads(put["Body"])["outcome"] for put in fake_boto3}
    assert outcomes == {"http_error", "response"}


def test_402_classified_as_credit_block(fake_boto3, trace_env):
    error = HTTPError(
        ot.OPENROUTER_CHAT_COMPLETIONS_URL,
        402,
        "Payment Required",
        {},
        io.BytesIO(b'{"error": "insufficient credits"}'),
    )
    with pytest.raises(ot.OpenRouterCreditBlockedError) as exc_info:
        _call(opener=_opener([error]))
    assert exc_info.value.http_status == 402
    assert exc_info.value.provider_usage["outcome"] == "http_error"


def test_transport_error_captured(fake_boto3, trace_env):
    with pytest.raises(ot.OpenRouterTelemetryError):
        _call(opener=_opener([URLError("connection refused")]))
    ot.flush_telemetry_traces()
    payload = json.loads(fake_boto3[0]["Body"])
    assert payload["outcome"] == "transport_error"


def test_streaming_refused_loudly():
    with pytest.raises(ot.OpenRouterStreamingUnsupportedError):
        _call(extra_body={"stream": True}, opener=_opener([]))


def test_no_s3_prefix_falls_back_to_private_model_manifest(fake_boto3, monkeypatch):
    monkeypatch.delenv(ot.RAW_TRACE_S3_PREFIX_ENV, raising=False)
    monkeypatch.setenv(
        ot.PRIVATE_MODEL_MANIFEST_URI_ENV,
        "s3://artifact-bucket/research-lab/sourcing-model/current.json",
    )
    monkeypatch.setenv(ot.RAW_TRACE_KMS_KEY_ENV, "kms-key-9")
    result = _call(opener=_opener([_chat_response()]))
    ot.flush_telemetry_traces()
    assert result.content == '{"ok": true}'
    assert result.raw_trace_ref is not None
    assert result.raw_trace_ref["s3_ref"].startswith(
        "s3://artifact-bucket/research-lab/sourcing-model/telemetry/"
    )
    assert result.provider_usage["reasoning_capture"]["storage_state"] == "raw_trace_ref"
    assert len(fake_boto3) == 1


def test_raw_trace_prefix_has_production_manifest_default(monkeypatch):
    monkeypatch.delenv(ot.RAW_TRACE_S3_PREFIX_ENV, raising=False)
    monkeypatch.delenv(ot.PRIVATE_MODEL_MANIFEST_URI_ENV, raising=False)

    assert ot.resolved_raw_trace_s3_prefix() == (
        "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model"
    )


def test_explicit_raw_trace_prefix_overrides_manifest(monkeypatch):
    monkeypatch.setenv(ot.RAW_TRACE_S3_PREFIX_ENV, "s3://explicit-bucket/raw/")
    monkeypatch.setenv(
        ot.PRIVATE_MODEL_MANIFEST_URI_ENV,
        "s3://artifact-bucket/research-lab/sourcing-model/current.json",
    )

    assert ot.resolved_raw_trace_s3_prefix() == "s3://explicit-bucket/raw"


def test_s3_prefix_without_kms_does_not_create_dangling_pointer(fake_boto3, monkeypatch):
    monkeypatch.setenv(ot.RAW_TRACE_S3_PREFIX_ENV, TRACE_PREFIX)
    monkeypatch.delenv(ot.RAW_TRACE_KMS_KEY_ENV, raising=False)
    result = _call(opener=_opener([_chat_response()]))
    ot.flush_telemetry_traces()
    assert result.content == '{"ok": true}'
    assert result.raw_trace_ref is None
    assert result.provider_usage["reasoning_capture"]["storage_state"] == "metadata_only"
    assert fake_boto3 == []


def test_record_openrouter_trace_hook(fake_boto3, trace_env):
    usage = ot.record_openrouter_trace(
        channel="qualification",
        purpose="intent_precheck",
        request_body={"model": "m", "messages": [], "include_reasoning": True},
        response_doc=_chat_response(reasoning="thinking"),
        context_ref="lead-1",
    )
    ot.flush_telemetry_traces()
    assert usage["raw_trace_ref"]["s3_ref"].startswith("s3://telemetry-bucket/")
    assert usage["reasoning_capture"]["returned"] is True
    assert usage["total_tokens"] == 19
    assert len(fake_boto3) == 1


def test_record_embeddings_hashes_only(fake_boto3, trace_env):
    doc = ot.record_openrouter_embeddings(
        channel="validator",
        purpose="stage5_verification",
        model_id="qwen/qwen3-embedding-8b",
        input_texts=["alpha", "beta"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )
    ot.flush_telemetry_traces()
    assert doc["input_count"] == 2
    assert all(h.startswith("sha256:") for h in doc["input_hashes"])
    assert doc["output_hash"].startswith("sha256:")
    payload = json.loads(fake_boto3[0]["Body"])
    # hashes only — raw vectors and raw input text never persisted
    assert "embeddings" not in payload
    assert "input_texts" not in payload
    assert "alpha" not in payload.values()
    assert payload["artifact_type"] == "research_lab_embeddings_trace"


def test_capture_failure_never_breaks_call(trace_env, monkeypatch):
    monkeypatch.setattr(ot, "_uploader", ot._TelemetryTraceUploader())

    def _boom(**kwargs):
        raise RuntimeError("capture exploded")

    monkeypatch.setattr(ot._uploader, "capture", _boom)
    result = _call(opener=_opener([_chat_response()]))
    assert result.content == '{"ok": true}'
