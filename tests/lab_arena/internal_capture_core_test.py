"""Host evidence uses existing storage without miner proofs or business effects."""
from __future__ import annotations

import json
import stat

import pytest

from lab_arena import internal_capture as capture, reasoning_capture, training_capture
from lab_arena.service import LocalObjectStore
from scripts.export_arena_internal_training import discover_s3, main


def request():
    return {"model": "test/model", "messages": [{"role": "user", "content": "Inspect the company."}],
            "reasoning": {"effort": "high", "exclude": False}, "max_tokens": 8192}


def response():
    return {"model": "test/model", "choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "Supported answer.", "reasoning_content": "Check the original source."}}],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 12}}}


def captured(tmp_path, *, provider=None, payload=None, cost=12, status=200, context=None, secrets=()):
    objects = LocalObjectStore(tmp_path)
    call = capture.InternalCapture(objects).start(purpose="admission_review", context=context or {
        "round_id": "arena-2026-11-05", "submission_id": "sub-example", "attempt": 1,
        "source_archive_sha256": "sha256:" + "a" * 64}, request=payload or request(), secrets=secrets)
    call.response(body=json.dumps(provider or response()).encode(), status=status, elapsed_seconds=0.25)
    call.finish(outcome={"host_decision": {"passed": True},
                         "committed_review": {"status": "deadline", "code_review_status": "error"}}, cost_microusd=cost)
    return objects, call


def test_roundtrip_retains_exact_request_readable_reasoning_and_distinct_outcomes(tmp_path):
    objects, call = captured(tmp_path)
    raw, candidate = capture.read_call(objects, call.prefix + "request.json")
    assert raw["completeness"]["complete"] is True
    assert raw["actor_type"] == candidate["actor_type"] == "arena_internal"
    assert raw["request"] == request()
    assert raw["response"] == response()
    assert raw["outcome"]["host_decision"]["passed"] is True
    assert raw["outcome"]["committed_review"]["code_review_status"] == "error"
    assert raw["events"]["request"]["dispatch"] == "not_established"
    assert candidate["target"] == {"role": "assistant", "content": "Supported answer."}
    assert candidate["generated_thinking"] == []  # Unreviewed route text is never full thinking.
    assert candidate["available_reasoning"][0]["text"] == "Check the original source."
    assert candidate["quality"] == {"status": "unreviewed", "correct_action": "not_established"}
    assert candidate["training_permission"]["status"] == "unconfirmed"
    assert "run_id" not in raw and "miner_hotkey" not in raw
    assert all(stage in raw["events"] for stage in capture.STAGES)


def test_start_is_not_dispatch_and_missing_stages_are_visible(tmp_path):
    objects = LocalObjectStore(tmp_path)
    call = capture.InternalCapture(objects).start(purpose="icp_generation", context={}, request=request())
    raw, candidate = capture.read_call(objects, call.prefix + "request.json")
    assert candidate is None
    assert raw["events"]["request"]["dispatch"] == "not_established"
    assert raw["completeness"]["dimensions"] == {
        "request": True, "response": False, "outcome": False, "privacy": True, "costs": False}
    assert "response_event_missing" in raw["completeness"]["gaps"]
    assert "outcome_event_missing" in raw["completeness"]["gaps"]


@pytest.mark.parametrize("cost", [None, -1, True, 1.2, "12"])
def test_unknown_cost_never_becomes_zero_or_a_known_charge(tmp_path, cost):
    objects, call = captured(tmp_path, cost=cost)
    raw, candidate = capture.read_call(objects, call.prefix + "request.json")
    assert raw["cost_status"] == "unknown" and raw["cost_microusd"] is None
    assert raw["completeness"]["complete"] is False
    assert candidate is not None  # Cost inventory is separate from observed target quality.
    assert candidate["source"]["cost_status"] == "unknown"


def test_known_zero_cost_is_preserved(tmp_path):
    objects, call = captured(tmp_path, cost=0)
    raw, _ = capture.read_call(objects, call.prefix + "request.json")
    assert raw["cost_status"] == "known" and raw["cost_microusd"] == 0


def test_secrets_are_filtered_from_bodies_context_and_outcome(tmp_path):
    secret = "host-private-example-secret"
    body = response()
    body["choices"][0]["message"]["reasoning_content"] += " " + secret
    payload = request()
    payload["messages"][0]["content"] += " " + secret
    objects, call = captured(tmp_path, provider=body, payload=payload,
                             context={"label": secret, "api_key": "other-private-value"}, secrets=(secret,))
    raw, candidate = capture.read_call(objects, call.prefix + "request.json")
    serialized = json.dumps(raw)
    assert secret not in serialized and "other-private-value" not in serialized
    assert raw["context"]["label"] == "[REDACTED]"
    assert raw["response"]["choices"][0]["message"]["reasoning_content"].endswith("[REDACTED]")
    assert raw["completeness"]["dimensions"]["privacy"] is False and candidate is None
    assert call._secrets == ()


def test_summary_and_opaque_state_remain_separate_from_full_text(tmp_path):
    provider = response()
    message = provider["choices"][0]["message"]
    message.pop("reasoning_content")
    message["reasoning_details"] = [{"type": "reasoning.summary", "summary": "Short summary."},
                                      {"type": "reasoning.encrypted", "data": "opaque-state"}]
    objects, call = captured(tmp_path, provider=provider)
    raw, candidate = capture.read_call(objects, call.prefix + "request.json")
    assert raw["reasoning"]["contains_summary"] is True
    assert raw["reasoning"]["contains_opaque"] is True
    assert raw["reasoning"]["full_generated_thinking"] == []
    assert "opaque-state" not in json.dumps(raw)
    assert candidate is None


def reviewed_registry():
    from tests.lab_arena.training_thinking_test import full_example
    return full_example()["configuration"]["reasoning_route_registry"]


def test_full_thinking_requires_approved_route_actual_provider_and_nonempty_text(tmp_path):
    registry = reviewed_registry()
    payload, annotation = reasoning_capture.apply("openrouter.chat", request(), policy=reasoning_capture.POLICY, registry=registry)
    provider = response()
    provider.update(provider="Route A", openrouter_metadata={"endpoints": {"available": [
        {"selected": True, "model": "test/model", "provider": "Route A"}]}})
    objects, call = captured(tmp_path, provider=provider, payload=payload)
    raw, candidate = capture.read_call(objects, call.prefix + "request.json", registry=registry)
    assert raw["reasoning"]["status"]["route_reviewed"] is True
    assert raw["reasoning"]["status"]["provider_route_verified"] is True
    assert candidate["generated_thinking"][0]["text"] == "Check the original source."

    provider["provider"] = "Different Provider"
    objects, call = captured(tmp_path / "mismatch", provider=provider, payload=payload)
    raw, candidate = capture.read_call(objects, call.prefix + "request.json", registry=registry)
    assert raw["reasoning"]["full_generated_thinking"] == []
    assert candidate["generated_thinking"] == []

    provider["provider"] = "Route A"
    provider["choices"][0]["message"].pop("reasoning_content")
    provider["choices"][0]["message"]["reasoning_details"] = [
        {"type": "reasoning.summary", "summary": "A summary is not the full text."}]
    objects, call = captured(tmp_path / "summary", provider=provider, payload=payload)
    raw, candidate = capture.read_call(objects, call.prefix + "request.json", registry=registry)
    assert candidate["generated_thinking"] == []


@pytest.mark.parametrize("finish", ["length", "content_filter", "error"])
def test_truncated_or_refused_results_remain_raw_without_sft_targets(tmp_path, finish):
    provider = response()
    provider["choices"][0]["finish_reason"] = finish
    objects, call = captured(tmp_path, provider=provider)
    raw, candidate = capture.read_call(objects, call.prefix + "request.json")
    assert raw["response"] == provider and candidate is None
    assert raw["sft_exclusions"] == ["target_shape_or_completion_unsupported"]


def test_native_search_tools_citations_and_evidence_are_retained_without_projection(tmp_path):
    objects = LocalObjectStore(tmp_path)
    payload = {"model": "test/model", "input": [{"role": "user", "content": "Find source."}],
               "tools": [{"type": "web_search_preview"}]}
    provider = {"object": "response", "status": "completed", "output": [
        {"id": "search-1", "type": "web_search_call", "status": "completed",
         "action": {"type": "search", "query": "company source"}},
        {"id": "reasoning-1", "type": "reasoning", "content": [
            {"type": "reasoning_text", "text": "Check the cited page."}]},
        {"id": "answer-1", "type": "message", "role": "assistant", "status": "completed", "content": [
            {"type": "output_text", "text": "Supported answer.", "annotations": [
                {"type": "url_citation", "url": "https://example.com/source", "title": "Original source",
                 "start_index": 0, "end_index": 9}]}]}]}
    call = capture.InternalCapture(objects).start(purpose="icp_generation", context={}, request=payload, api="openrouter.responses")
    call.response(body=json.dumps(provider).encode(), status=200)
    call.finish(outcome={"committed_outcome": "activated"}, cost_microusd=1)
    raw, _ = capture.read_call(objects, call.prefix + "request.json")
    assert raw["request"] == payload and raw["response"] == provider
    assert raw["reasoning"]["available_segments"][0]["text"] == "Check the cited page."


@pytest.mark.parametrize("body", [b"provider returned plain text", b"\xff\xfe", b"{broken"])
def test_nonjson_and_unsupported_encoding_are_explicit_and_nonthrowing(tmp_path, body):
    objects = LocalObjectStore(tmp_path)
    call = capture.InternalCapture(objects).start(purpose="icp_exclusion", context={}, request=request())
    call.response(body=body, status=503)
    call.finish(outcome={"business_outcome": "failed"})
    raw, candidate = capture.read_call(objects, call.prefix + "request.json")
    assert candidate is None and raw["events"]["response"]["http_status"] == 503
    if body != b"\xff\xfe":
        assert raw["response_text"] == body.decode()
    else:
        assert "response_body_unavailable_or_invalid" in raw["completeness"]["gaps"]


def test_all_capture_storage_failures_are_observational():
    class Broken:
        def put(self, *args):
            raise OSError("private diagnostic that must not escape")
    call = capture.InternalCapture(Broken()).start(purpose="admission_review", context={}, request=request())
    assert call.call_id and call.request_pointer["status"] == "unavailable"
    assert call.response(body=b"{}", status=200)["status"] == "unavailable"
    assert call.finish(outcome={"committed_review": "passed"})["status"] == "unavailable"
    disabled = capture.InternalCapture(None).start(purpose="icp_exclusion", context={}, request=request())
    disabled.response(body=b"{}", status=200)
    disabled.finish(outcome={})


def test_index_outage_logs_only_fixed_metadata_without_secret_or_error_text(caplog):
    secret = "never-log-this-private-credential"
    class Broken:
        def put(self, *args):
            raise OSError(secret + " storage endpoint and private request")
    call = capture.InternalCapture(Broken()).start(
        purpose="admission_review", context={"private_context": secret}, request=request(), secrets=(secret,))
    response_result = call.response(body=json.dumps(response()).encode(), status=200)
    finish_result = call.finish(outcome={"committed_review": {"status": "passed"}, "private_note": secret}, cost_microusd=12)
    assert response_result["status"] == finish_result["status"] == "unavailable"
    messages = [record.getMessage() for record in caplog.records if record.name == capture.__name__]
    assert messages == ["arena_internal_capture_write_failed stage=" + stage
                        + " purpose=admission_review call_id=" + call.call_id for stage in capture.STAGES]
    assert secret not in caplog.text
    assert "storage endpoint" not in caplog.text and "private_context" not in caplog.text
    assert "private_note" not in caplog.text and "Supported answer" not in caplog.text


def test_unknown_purpose_and_logger_failure_cannot_expose_data_or_break_capture(monkeypatch, caplog):
    call = capture.InternalCapture(None).start(purpose="private_label", context={}, request=request())
    assert "purpose=unknown" in caplog.text and "private_label" not in caplog.text
    def failed_logger(*args, **kwargs):
        raise RuntimeError("logging unavailable")
    monkeypatch.setattr(capture._LOGGER, "warning", failed_logger)
    call.response(body=b"{}", status=200)
    assert call.finish(outcome={"status": "unchanged"})["status"] == "unavailable"


def test_duplicate_stage_cannot_overwrite_original_evidence(tmp_path):
    objects, call = captured(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    call.response(body=b'{"replacement":true}', status=500)
    call.finish(outcome={"replacement": "forbidden"}, cost_microusd=999)
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_tampered_event_or_body_is_a_gap_and_not_a_target(tmp_path):
    objects, call = captured(tmp_path)
    path = tmp_path / (call.prefix + "response.json")
    event = json.loads(path.read_text())
    event["http_status"] = 500
    path.write_text(json.dumps(event))
    raw, candidate = capture.read_call(objects, call.prefix + "request.json")
    assert candidate is None
    assert "response_event_unavailable_or_invalid" in raw["completeness"]["gaps"]


@pytest.mark.parametrize("ref", ["/etc/passwd", "../private/request.json",
    "private/arena-training/internal/2026-02-31/" + "a" * 32 + "/request.json",
    "private/arena-training/internal/2026-11-05/" + "a" * 32 + "/other.json"])
def test_arbitrary_or_invalid_refs_are_rejected_before_read(ref):
    class NoRead:
        def get_bounded(self, *args):
            pytest.fail("invalid reference reached object storage")
    with pytest.raises(ValueError):
        capture.read_call(NoRead(), ref)


def test_cli_outputs_private_files_and_refuses_overwrite(tmp_path):
    objects, call = captured(tmp_path / "objects")
    output = tmp_path / "export"
    args = ["--object-root", str(tmp_path / "objects"), "--refs", call.prefix + "request.json", "--output", str(output)]
    assert main(args) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"] == {"calls": 1, "complete": 1, "sft_candidates": 1}
    assert manifest["discovery"]["method"] == "explicit_refs"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert main(args) == 1
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_s3_discovery_is_date_scoped_bounded_and_reports_truncation():
    class Client:
        def __init__(self):
            self.calls = []
        def list_objects_v2(self, **kwargs):
            self.calls.append(kwargs)
            return {"Contents": [{"Key": kwargs["Prefix"] + "a" * 32 + "/request.json"}],
                    "IsTruncated": True, "NextContinuationToken": "next"}
    client = Client()
    refs, summary = discover_s3(client, bucket="private-bucket", prefix="arena", dates=["2026-11-05"], max_events=1)
    assert len(refs) == 1 and summary["truncated"] is True
    assert client.calls == [{"Bucket": "private-bucket", "Prefix": "arena/private/arena-training/internal/2026-11-05/", "MaxKeys": 1}]
    with pytest.raises(ValueError):
        discover_s3(client, bucket="private", prefix="", dates=[], max_events=1)


def test_s3_discovery_follows_only_same_date_cursor_and_dedupes_refs():
    class Client:
        def __init__(self):
            self.calls = []
        def list_objects_v2(self, **kwargs):
            self.calls.append(kwargs)
            return {"Contents": [{"Key": kwargs["Prefix"] + "a" * 32 + "/" + (
                "response.json" if kwargs.get("ContinuationToken") else "request.json")}],
                "IsTruncated": len(self.calls) == 1, "NextContinuationToken": "next"}
    client = Client()
    refs, summary = discover_s3(client, bucket="private", prefix="", dates=["2026-11-05"], max_events=10)
    assert len(refs) == 2 and summary["truncated"] is False
    assert len(client.calls) == 2 and client.calls[1]["ContinuationToken"] == "next"
    assert client.calls[0]["Prefix"] == client.calls[1]["Prefix"]


def test_environment_unavailable_returns_disabled_sink(monkeypatch):
    monkeypatch.delenv("LAB_ARENA_BUCKET", raising=False)
    sink = capture.from_environment()
    assert isinstance(sink, capture.InternalCapture) and sink.objects is None


def test_environment_uses_same_bucket_prefix_and_short_no_retry_client(monkeypatch):
    import boto3
    observed = []
    fake_client = object()
    def client(*args, **kwargs):
        observed.append((args, kwargs))
        return fake_client
    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setenv("LAB_ARENA_BUCKET", "existing-private-bucket")
    monkeypatch.setenv("LAB_ARENA_OBJECT_PREFIX", "arena")
    sink = capture.from_environment()
    assert sink.objects._bucket == "existing-private-bucket"
    assert sink.objects._key_prefix == "arena" and sink.objects._client is fake_client
    options = observed[0][1]["config"]
    assert options.connect_timeout == options.read_timeout == 0.5
    assert options.retries == {"total_max_attempts": 1, "mode": "standard"}


def test_oversized_request_and_outcome_leave_small_explicit_gaps(tmp_path):
    objects = LocalObjectStore(tmp_path)
    payload = request()
    payload["messages"][0]["content"] = "x" * training_capture.MAX_OBJECT_BYTES
    call = capture.InternalCapture(objects).start(purpose="icp_generation", context={}, request=payload)
    call.response(body=b"{}", status=200)
    call.finish(outcome={"value": "x" * capture.MAX_OUTCOME_BYTES})
    raw, candidate = capture.read_call(objects, call.prefix + "request.json")
    assert raw["purpose"] == "icp_generation"
    assert raw["events"]["request"]["capture"]["reason"] == "request_size_limit"
    assert raw["events"]["outcome"]["capture"]["reason"] == "internal_outcome_capture_failed"
    assert raw["completeness"]["dimensions"]["request"] is False
    assert raw["completeness"]["dimensions"]["outcome"] is False and candidate is None
