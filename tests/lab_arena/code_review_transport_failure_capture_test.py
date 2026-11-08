"""Incomplete admission transport bodies stay private evidence, never a target."""
import json

import httpx
import pytest

from lab_arena import broker, code_review, contracts, internal_capture
from lab_arena.service import LocalObjectStore
from scripts.export_arena_internal_training import main as export_internal
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, ReviewTransport, _archive, _review_body, _reviewer, _source,
)
from tests.lab_arena.transport_failure_body_capture_test import transport_fixture


class TrackedCapture(internal_capture.InternalCapture):
    def __init__(self, objects):
        super().__init__(objects)
        self.calls = []

    def start(self, **kwargs):
        call = super().start(**kwargs)
        self.calls.append(call)
        return call


def _run(tmp_path, payload, *, status=200, failure=httpx.ReadTimeout):
    transport, stream, sent = transport_fixture(payload, failure, status)
    objects = LocalObjectStore(tmp_path / "objects")
    capture = TrackedCapture(objects)
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    reviewer._capture = capture
    try:
        result = reviewer.review(row)
    finally:
        transport.close()
    assert len(sent) == len(capture.calls) == 1 and stream.closed
    assert result["status"] == "error"
    assert len(store.finish_calls) == 1
    assert store.finish_calls[0][4]["error_code"] == "code_review_transport_error"
    assert store.finish_calls[0][5] is None
    call = capture.calls[0]
    raw, candidate = internal_capture.read_call(objects, call.prefix + "request.json")
    return raw, candidate, objects, call, store


def _complete_body():
    return json.dumps({"model": code_review.DEFAULT_REVIEW_MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "Safe fixture answer.",
            "reasoning": "Read the submitted source."}}],
        "usage": {"cost": "0.000123", "completion_tokens_details": {"reasoning_tokens": 4}}},
        separators=(",", ":")).encode()


@pytest.mark.parametrize("status", [200, 503])
def test_real_timeout_retains_exact_valid_json_status_elapsed_without_verdict_or_sft(tmp_path, status):
    body = _complete_body()
    raw, candidate, objects, call, store = _run(tmp_path, body, status=status)
    assert raw["response"] == json.loads(body)
    assert raw["events"]["response"]["http_status"] == status
    assert raw["events"]["response"]["wire_body_sha256"] == contracts.hash_bytes(body)
    assert raw["events"]["response"]["transport_completed"] is False
    assert raw["events"]["response"]["elapsed_seconds"] >= 0
    assert raw["outcome"]["host_decision"]["error_code"] == "code_review_transport_error"
    assert raw["cost_status"] == "unknown" and candidate is None
    assert "transport_incomplete" in raw["completeness"]["gaps"]
    assert raw["sft_exclusions"] == ["transport_incomplete"]
    assert "passed" not in store.finish_calls[0][4]
    output = tmp_path / "export"
    assert export_internal(["--refs", call.prefix + "request.json",
        "--object-root", str(tmp_path / "objects"), "--output", str(output)]) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"].get("sft_candidates", 0) == 0
    assert (output / "sft.jsonl").read_text() == ""


@pytest.mark.parametrize("payload", [
    b"", b" ", b'{"choices":[', b'{"choices":[],"choices":[]}',
    b"[]", b'{"value":NaN}', b'{"value":Infinity}',
    b'{"choices":[]}'+b" "*(3*1024*1024),
    b'{"text":"' + "".join("\\u%04x" % ord(c) for c in MINER_KEY).encode() + b'"}',
])
def test_invalid_or_unsafe_timeout_body_has_no_private_response_stage(tmp_path, payload):
    raw, candidate, objects, call, store = _run(tmp_path, payload)
    assert "response" not in raw["events"]
    assert raw["response"] is None and candidate is None
    assert "response_event_missing" in raw["completeness"]["gaps"]
    saved = b"\n".join(path.read_bytes() for path in (tmp_path / "objects").rglob("*") if path.is_file())
    assert MINER_KEY.encode() not in saved
    assert "response" not in call._stages
    assert store.finish_calls[0][5] is None


def test_other_private_fields_use_existing_structured_sanitizer(tmp_path):
    body = json.dumps({"api_key": "private-other-key", "choices": [{
        "index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "Safe fixture answer.",
            "encrypted_content": "private-opaque-state", "reasoning": "Safe thought."}}]}).encode()
    raw, candidate, objects, call, _ = _run(tmp_path, body)
    saved = b"\n".join(path.read_bytes() for path in (tmp_path / "objects").rglob("*") if path.is_file())
    assert b"private-other-key" not in saved and b"private-opaque-state" not in saved
    assert raw["response"] is not None and candidate is None
    assert raw["events"]["response"]["capture"]["redacted"] is True
    assert raw["events"]["response"]["transport_completed"] is False
    assert "transport_incomplete" in raw["completeness"]["gaps"]


def test_completed_http200_response_keeps_existing_candidate_even_when_committed_error(tmp_path):
    objects = LocalObjectStore(tmp_path / "objects")
    capture = TrackedCapture(objects)
    transport = ReviewTransport(lambda parameters: {"body": _review_body(parameters)})
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    reviewer._capture = capture
    assert reviewer.review(row)["status"] == "passed"
    call = capture.calls[0]
    raw, candidate = internal_capture.read_call(objects, call.prefix + "request.json")
    assert candidate is not None
    assert "transport_completed" not in raw["events"]["response"]
    assert raw["completeness"]["dimensions"]["response"] is True

    # Historical records with a completed 200 and a later error outcome remain
    # candidate inventory. The new guard applies only to an explicit timeout.
    second = capture.start(purpose="admission_review", context={}, request=raw["request"], api="openrouter.chat")
    second.response(body=_complete_body(), status=200)
    second.finish(outcome={"host_decision": {"error_code": "code_review_transport_error"},
                           "committed_review": {"status": "error"}}, cost_microusd=None)
    old_raw, old_candidate = internal_capture.read_call(objects, second.prefix + "request.json")
    assert old_candidate is not None
    assert "transport_completed" not in old_raw["events"]["response"]
