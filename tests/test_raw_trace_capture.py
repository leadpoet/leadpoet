"""Tests for §9.1 item 5 — raw prompt/response capture at the OpenRouter boundary.

Covers:
  * KMS-encrypted S3 writes under
    ``{prefix}/trajectories/{run_id}/{stage}/{seq}-{stage}.json.enc`` with the
    trace encryption KMS key id (SSE-KMS headers on put_object).
  * Pointer-not-content: loop-event / provider-usage docs carry only
    ``{s3_ref, sha256}`` and pass the trajectory-corpus protected-material scan.
  * Credential redaction: Authorization / api keys never reach the trace store;
    key-shaped strings inside message content are redacted.
  * Absolute safety rail: an S3/KMS/boto3 failure never fails or changes the
    LLM call result, and failure logs once per run (not per call).
  * Inert paths: flag off, missing bucket config (one log then silent),
    missing boto3 (process-wide disable after one log), no capture_run_id.
  * Failed attempts are training signal: HTTP-error, transport-error, and
    length-retry attempts are each captured, and length-retry pointers ride
    ``retry_attempts`` in the final provider_usage.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import types
from urllib.error import HTTPError, URLError

import pytest

import gateway.research_lab.worker as worker_mod
from gateway.research_lab.config import ResearchLabGatewayConfig
from research_lab.canonical import sha256_bytes
from research_lab.trajectory_corpus import _find_protected_corpus_material


TRACE_PREFIX = "s3://test-trace-bucket/lab-traces"
RUN_ID = "run-trace-1"
PROMPT_TEXT = "def source_provider():  # private repo excerpt for the draft"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeS3Client:
    def __init__(self, puts, fail_with=None):
        self._puts = puts
        self._fail_with = fail_with

    def put_object(self, **kwargs):
        if self._fail_with is not None:
            raise self._fail_with
        self._puts.append(kwargs)


@pytest.fixture
def fake_boto3(monkeypatch):
    puts: list[dict] = []
    fake = types.ModuleType("boto3")
    fake.client = lambda name, **kwargs: _FakeS3Client(puts)
    monkeypatch.setitem(sys.modules, "boto3", fake)
    return puts


@pytest.fixture
def trace_env(monkeypatch):
    monkeypatch.setenv(worker_mod._RAW_TRACE_S3_PREFIX_ENV, TRACE_PREFIX)
    monkeypatch.setenv(worker_mod._TRACE_KMS_KEY_ENV, "trace-kms-key-1")
    monkeypatch.delenv(worker_mod._RAW_TRACE_CAPTURE_ENABLED_ENV, raising=False)


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(responses):
    """Sequential fake for urllib.request.urlopen; Exception items are raised."""

    def opener(req, timeout=0):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeHTTPResponse(item)

    return opener


def _chat_response(content='{"ok": true}', finish_reason="stop"):
    # No "id": keeps _build_openrouter_provider_usage off the generation-stats
    # HTTP path (cost_reconciliation_status=missing_response_id).
    return {
        "model": "test/model",
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish_reason,
                "native_finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
    }


def _worker():
    return worker_mod.ResearchLabHostedWorker(ResearchLabGatewayConfig(), worker_ref="worker-trace")


def _call(worker, *, capture_run_id=RUN_ID, capture_stage="code_edit_draft"):
    return asyncio.run(
        worker._call_openrouter(
            messages=[{"role": "user", "content": PROMPT_TEXT}],
            api_key="sk-or-test-key-123",
            model_id="test/model",
            timeout_seconds=5,
            max_tokens=100,
            capture_run_id=capture_run_id,
            capture_stage=capture_stage,
        )
    )


# ---------------------------------------------------------------------------
# recorder: key scheme, SSE-KMS, redaction, seq ordering
# ---------------------------------------------------------------------------


def test_recorder_writes_kms_encrypted_object_with_pointer(fake_boto3, trace_env):
    config = ResearchLabGatewayConfig()
    recorder = worker_mod._OpenRouterRawTraceRecorder(config)
    ref = recorder.capture(
        run_id="run-abc",
        stage="loop_planner",
        request_doc={
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "method": "POST",
            "body": {"messages": [{"role": "user", "content": "inline sk-or-v1-deadbeef leak"}]},
        },
        response_doc={"choices": [{"message": {"content": "done"}}]},
        outcome="response",
    )
    recorder.flush()

    assert ref is not None
    assert set(ref) == {"s3_ref", "sha256"}
    assert len(fake_boto3) == 1
    put = fake_boto3[0]
    assert put["Bucket"] == "test-trace-bucket"
    assert put["Key"] == "lab-traces/trajectories/run-abc/loop_planner/0001-loop_planner.json.enc"
    assert put["ServerSideEncryption"] == "aws:kms"
    assert put["SSEKMSKeyId"] == "trace-kms-key-1"
    assert put["ContentType"] == "application/json"
    assert ref["s3_ref"] == f"s3://test-trace-bucket/{put['Key']}"
    # sha256 pointer verifies the exact stored bytes.
    assert ref["sha256"] == sha256_bytes(put["Body"])

    payload = json.loads(put["Body"])
    assert payload["artifact_type"] == "research_lab_raw_llm_trace"
    assert payload["run_id"] == "run-abc"
    assert payload["stage"] == "loop_planner"
    assert payload["seq"] == 1
    assert payload["outcome"] == "response"
    body_text = put["Body"].decode("utf-8")
    assert "sk-or-" not in body_text
    assert "[redacted-openrouter-key]" in body_text


def test_recorder_stamps_derived_axis_provenance(fake_boto3, trace_env):
    """P11: every persisted trace carries derived call_emitter / purpose /
    teacher_model_flag, and captures inside a call_episode scope carry the
    episode correlation id."""
    from research_lab.axis_provenance import call_episode

    recorder = worker_mod._OpenRouterRawTraceRecorder(ResearchLabGatewayConfig())
    request_doc = {"url": "u", "method": "POST", "body": {}}
    recorder.capture(
        run_id="run-abc", stage="loop_planner", request_doc=request_doc,
        response_doc={}, outcome="response",
    )
    recorder.capture(
        run_id="run-abc", stage="plan_alignment_judge", request_doc=request_doc,
        response_doc={}, outcome="response",
    )
    with call_episode(run_id="run-abc", iteration=2, inspection_round=3):
        recorder.capture(
            run_id="run-abc", stage="source_inspection", request_doc=request_doc,
            response_doc={}, outcome="response",
        )
    recorder.flush()

    payloads = {p["stage"]: p for p in (json.loads(put["Body"]) for put in fake_boto3)}
    planner = payloads["loop_planner"]
    assert planner["call_emitter"] == "code"
    assert planner["purpose"] == "plan_next_iteration"
    assert planner["teacher_model_flag"] is False
    assert "episode" not in planner

    judge = payloads["plan_alignment_judge"]
    assert judge["call_emitter"] == "code"
    assert judge["teacher_model_flag"] is True

    inspection = payloads["source_inspection"]
    assert inspection["call_emitter"] == "model"
    assert inspection["episode"] == {
        "run_id": "run-abc", "iteration": 2, "inspection_round": 3,
    }
    assert inspection["episode_id"] == "run-abc:i2:r3"


def test_recorder_seq_is_global_per_run_across_stages(fake_boto3, trace_env):
    recorder = worker_mod._OpenRouterRawTraceRecorder(ResearchLabGatewayConfig())
    request_doc = {"url": "u", "method": "POST", "body": {}}
    recorder.capture(run_id="run-abc", stage="loop_planner", request_doc=request_doc, response_doc={}, outcome="response")
    recorder.capture(run_id="run-abc", stage="code_edit_draft", request_doc=request_doc, response_doc={}, outcome="response")
    recorder.capture(run_id="run-other", stage="loop_planner", request_doc=request_doc, response_doc={}, outcome="response")
    recorder.flush()
    keys = sorted(put["Key"] for put in fake_boto3)
    assert keys == [
        "lab-traces/trajectories/run-abc/code_edit_draft/0002-code_edit_draft.json.enc",
        "lab-traces/trajectories/run-abc/loop_planner/0001-loop_planner.json.enc",
        "lab-traces/trajectories/run-other/loop_planner/0001-loop_planner.json.enc",
    ]


def test_recorder_falls_back_to_manifest_uri_prefix(fake_boto3, monkeypatch):
    monkeypatch.delenv(worker_mod._RAW_TRACE_S3_PREFIX_ENV, raising=False)
    monkeypatch.setenv(worker_mod._TRACE_KMS_KEY_ENV, "trace-kms-key-1")
    config = ResearchLabGatewayConfig(
        private_model_manifest_uri="s3://artifact-bucket/research-lab/sourcing-model/current.json"
    )
    recorder = worker_mod._OpenRouterRawTraceRecorder(config)
    ref = recorder.capture(
        run_id="run-abc",
        stage="loop_planner",
        request_doc={"url": "u", "method": "POST", "body": {}},
        response_doc={},
        outcome="response",
    )
    recorder.flush()
    assert ref is not None
    assert fake_boto3[0]["Bucket"] == "artifact-bucket"
    assert fake_boto3[0]["Key"] == (
        "research-lab/sourcing-model/trajectories/run-abc/loop_planner/0001-loop_planner.json.enc"
    )


# ---------------------------------------------------------------------------
# call boundary: pointer-not-content in provider-usage / event docs
# ---------------------------------------------------------------------------


def test_call_attaches_pointer_not_content(fake_boto3, trace_env, monkeypatch):
    worker = _worker()
    monkeypatch.setattr(worker_mod.urlrequest, "urlopen", _fake_urlopen([_chat_response(content='{"plan": "x"}')]))
    result = _call(worker)
    worker._raw_trace_recorder.flush()

    assert result.content == '{"plan": "x"}'
    ref = result.provider_usage["raw_trace_ref"]
    assert set(ref) == {"s3_ref", "sha256"}

    # Shaped like a loop-event doc: pointers only, no raw prompt/response text,
    # and the trajectory-corpus protected-material scanner finds nothing.
    event_doc_shaped = {"provider_usage": [dict(result.provider_usage)]}
    serialized = json.dumps(event_doc_shaped)
    assert PROMPT_TEXT not in serialized
    assert '{"plan": "x"}' not in serialized
    assert "sk-or-test-key-123" not in serialized
    assert _find_protected_corpus_material(event_doc_shaped) == set()

    # The encrypted S3 object, by contrast, holds the full request + response.
    assert len(fake_boto3) == 1
    payload = json.loads(fake_boto3[0]["Body"])
    assert payload["request"]["body"]["messages"][0]["content"] == PROMPT_TEXT
    assert payload["response"]["choices"][0]["message"]["content"] == '{"plan": "x"}'
    # The miner's OpenRouter key (Authorization header) never reaches the store.
    body_text = fake_boto3[0]["Body"].decode("utf-8")
    assert "sk-or-test-key-123" not in body_text
    assert "Authorization" not in body_text


def test_call_without_run_id_captures_nothing(fake_boto3, trace_env, monkeypatch):
    worker = _worker()
    monkeypatch.setattr(worker_mod.urlrequest, "urlopen", _fake_urlopen([_chat_response()]))
    result = _call(worker, capture_run_id="")
    worker._raw_trace_recorder.flush()
    assert result.content == '{"ok": true}'
    assert "raw_trace_ref" not in result.provider_usage
    assert fake_boto3 == []


def test_flag_disables_capture(fake_boto3, trace_env, monkeypatch):
    monkeypatch.setenv(worker_mod._RAW_TRACE_CAPTURE_ENABLED_ENV, "false")
    worker = _worker()
    monkeypatch.setattr(worker_mod.urlrequest, "urlopen", _fake_urlopen([_chat_response()]))
    result = _call(worker)
    worker._raw_trace_recorder.flush()
    assert result.content == '{"ok": true}'
    assert "raw_trace_ref" not in result.provider_usage
    assert fake_boto3 == []


# ---------------------------------------------------------------------------
# failed attempts are captured too (training signal)
# ---------------------------------------------------------------------------


def test_length_retry_attempts_each_captured(fake_boto3, trace_env, monkeypatch):
    worker = _worker()
    monkeypatch.setattr(
        worker_mod.urlrequest,
        "urlopen",
        _fake_urlopen(
            [
                _chat_response(content="partial", finish_reason="length"),
                _chat_response(content='{"ok": true}'),
            ]
        ),
    )
    result = _call(worker)
    worker._raw_trace_recorder.flush()

    assert result.content == '{"ok": true}'
    assert len(fake_boto3) == 2
    keys = [put["Key"] for put in fake_boto3]
    assert keys[0].endswith("/0001-code_edit_draft.json.enc")
    assert keys[1].endswith("/0002-code_edit_draft.json.enc")

    # The failed length attempt's pointer rides retry_attempts; the final
    # response carries its own pointer.
    retry_attempts = result.provider_usage["retry_attempts"]
    assert retry_attempts[0]["raw_trace_ref"]["s3_ref"] == f"s3://test-trace-bucket/{keys[0]}"
    assert result.provider_usage["raw_trace_ref"]["s3_ref"] == f"s3://test-trace-bucket/{keys[1]}"
    first_payload = json.loads(fake_boto3[0]["Body"])
    assert first_payload["response"]["choices"][0]["finish_reason"] == "length"


def test_http_error_attempt_captured(fake_boto3, trace_env, monkeypatch):
    error = HTTPError(
        "https://openrouter.ai/api/v1/chat/completions",
        500,
        "boom",
        None,
        io.BytesIO(b"upstream exploded"),
    )
    worker = _worker()
    monkeypatch.setattr(worker_mod.urlrequest, "urlopen", _fake_urlopen([error, _chat_response()]))
    result = _call(worker)
    worker._raw_trace_recorder.flush()

    assert result.content == '{"ok": true}'
    assert len(fake_boto3) == 2
    first_payload = json.loads(fake_boto3[0]["Body"])
    assert first_payload["outcome"] == "http_error"
    assert first_payload["response"]["http_status"] == 500
    assert first_payload["response"]["error_excerpt"] == "upstream exploded"
    # Failed attempts keep the prompt: that is the training signal.
    assert first_payload["request"]["body"]["messages"][0]["content"] == PROMPT_TEXT


def test_transport_error_attempt_captured_with_proxy_redaction(fake_boto3, trace_env, monkeypatch):
    error = URLError("proxy http://user:secret@proxy.example.com refused")
    worker = _worker()
    monkeypatch.setattr(worker_mod.urlrequest, "urlopen", _fake_urlopen([error, _chat_response()]))
    result = _call(worker)
    worker._raw_trace_recorder.flush()

    assert result.content == '{"ok": true}'
    assert len(fake_boto3) == 2
    first_payload = json.loads(fake_boto3[0]["Body"])
    assert first_payload["outcome"] == "transport_error"
    excerpt = first_payload["response"]["error_excerpt"]
    assert "user:secret" not in excerpt
    assert "[redacted-proxy-url]@proxy.example.com" in excerpt


# ---------------------------------------------------------------------------
# absolute safety rail: capture failure never touches the LLM call
# ---------------------------------------------------------------------------


def test_s3_put_failure_keeps_call_result_and_logs_once_per_run(trace_env, monkeypatch, caplog):
    fake = types.ModuleType("boto3")
    fake.client = lambda name, **kwargs: _FakeS3Client([], fail_with=RuntimeError("kms denied"))
    monkeypatch.setitem(sys.modules, "boto3", fake)
    worker = _worker()
    monkeypatch.setattr(
        worker_mod.urlrequest,
        "urlopen",
        _fake_urlopen([_chat_response(content="one"), _chat_response(content="two")]),
    )
    with caplog.at_level(logging.WARNING, logger="gateway.research_lab.worker"):
        first = _call(worker)
        second = _call(worker)
        worker._raw_trace_recorder.flush()

    # LLM call results are completely unaffected by the failing writes.
    assert first.content == "one"
    assert second.content == "two"
    # Pointers stay attached (optimistic; documented as dangling-never-wrong).
    assert "raw_trace_ref" in first.provider_usage
    assert "raw_trace_ref" in second.provider_usage
    failures = [m for m in caplog.messages if m.startswith("research_lab_raw_trace_capture_failed")]
    assert len(failures) == 1
    assert "kms denied" in failures[0]


def test_missing_boto3_disables_capture_after_one_log(trace_env, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "boto3", None)  # import boto3 -> ImportError
    worker = _worker()
    monkeypatch.setattr(
        worker_mod.urlrequest,
        "urlopen",
        _fake_urlopen([_chat_response(content="one"), _chat_response(content="two")]),
    )
    with caplog.at_level(logging.WARNING, logger="gateway.research_lab.worker"):
        first = _call(worker)
        worker._raw_trace_recorder.flush()
        second = _call(worker)
        worker._raw_trace_recorder.flush()

    assert first.content == "one"
    assert second.content == "two"
    # Recorder went inert after the import failure: the second call did not
    # even attach a pointer, and only one failure line was logged.
    assert "raw_trace_ref" not in second.provider_usage
    failures = [m for m in caplog.messages if m.startswith("research_lab_raw_trace_capture_failed")]
    assert len(failures) == 1


def test_missing_bucket_config_skips_after_one_log(monkeypatch, caplog):
    monkeypatch.delenv(worker_mod._RAW_TRACE_S3_PREFIX_ENV, raising=False)
    config = ResearchLabGatewayConfig(private_model_manifest_uri="")
    recorder = worker_mod._OpenRouterRawTraceRecorder(config)
    request_doc = {"url": "u", "method": "POST", "body": {}}
    with caplog.at_level(logging.INFO, logger="gateway.research_lab.worker"):
        assert recorder.capture(run_id="run-a", stage="s", request_doc=request_doc, response_doc={}, outcome="response") is None
        assert recorder.capture(run_id="run-a", stage="s", request_doc=request_doc, response_doc={}, outcome="response") is None
        assert recorder.capture(run_id="run-b", stage="s", request_doc=request_doc, response_doc={}, outcome="response") is None
    disabled_logs = [m for m in caplog.messages if m.startswith("research_lab_raw_trace_capture_unavailable")]
    assert len(disabled_logs) == 1
    assert "reason=missing_s3_prefix" in disabled_logs[0]
    # Fully inert: no background pool was ever spawned.
    assert recorder._executor is None


def test_non_s3_prefix_override_is_inert(monkeypatch, caplog):
    monkeypatch.setenv(worker_mod._RAW_TRACE_S3_PREFIX_ENV, "file:///tmp/traces")
    recorder = worker_mod._OpenRouterRawTraceRecorder(ResearchLabGatewayConfig())
    with caplog.at_level(logging.INFO, logger="gateway.research_lab.worker"):
        ref = recorder.capture(
            run_id="run-a",
            stage="s",
            request_doc={"url": "u", "method": "POST", "body": {}},
            response_doc={},
            outcome="response",
        )
    assert ref is None
    assert recorder._executor is None


# ---------------------------------------------------------------------------
# redaction helper
# ---------------------------------------------------------------------------


def test_redact_raw_trace_value_scrubs_credentials_but_keeps_prose():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "the proxy password handling in sourcing uses "
                    "sk-or-v1-abc123 and AKIAABCDEFGHIJKLMNOP via "
                    "https://user:pw@proxy.example.com/route?api_key=topsecret&x=1"
                ),
            }
        ],
        "nested": ["sb_secret_zzz", {"deep": "sb_publishable_yyy"}],
    }
    redacted = worker_mod._redact_raw_trace_value(payload)
    text = json.dumps(redacted)
    assert "sk-or-v1-abc123" not in text
    assert "AKIAABCDEFGHIJKLMNOP" not in text
    assert "user:pw@" not in text
    assert "api_key=topsecret" not in text
    assert "sb_secret_zzz" not in text
    assert "sb_publishable_yyy" not in text
    # Unlike the diagnostic-text nuke, ordinary prose mentioning secret-ish
    # words survives — the capture exists to preserve prompt text.
    assert "the proxy password handling in sourcing" in text
