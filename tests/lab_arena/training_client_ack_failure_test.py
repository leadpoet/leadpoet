"""Optional read-report failures cannot invalidate exact host-delivered results."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import pytest

from lab_arena import automatic_capture, contracts, lab_arena_checkpoint as checkpoint, training_eligibility, training_observations as observations, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import submission
from tests.lab_arena.training_optional_capture_test import captured_timeout, refresh
from tests.lab_arena.training_worker_proof_reuse_test import large_example


def captured_failure(*, category="response_received", reason="transport_error", damage=None,
                     failed_writes=0, runtime_summary=None):
    template = captured_timeout()
    example = large_example(20) if failed_writes else deepcopy(template)
    if failed_writes:
        for key in ("completeness", "attempts", "related_run_gap_attribution", "score_run_id",
                    "input_snapshots", "output", "output_identity", "judgment"):
            example[key] = deepcopy(template[key])
    artifact = example["worker_observation_artifacts"]["run"]
    buffer = observations.Buffer()
    for index, receipt in enumerate(artifact["payload"]["response_receipts"]):
        receipt = deepcopy(receipt)
        if index < failed_writes:
            receipt.update(write_state="socket_write_failed", socket_written_at=None)
            example["provider_calls"][index]["response_provenance"]["delivery_to_worker"] = "socket_write_failed"
        assert buffer.append_response_receipt(receipt)
    row = deepcopy(template["worker_observation_artifacts"]["run"]["payload"]["agent_observations"][0])
    if runtime_summary is not None:
        row["data"]["value"] = deepcopy(runtime_summary)
    value = row["data"]["value"]
    if runtime_summary is None:
        value["counts"] = {"responses_started": 1, "responses_complete": 1, "client_httpx": 1,
                           "checkpoint_failure": 1, "checkpoint_failure_" + category + "_" + reason: 1}
    if damage in {"pending_queue", "inflight"}:
        value[damage] = 1
    elif damage == "unbalanced":
        value["counts"]["responses_started"] += 1
    elif damage == "unknown_summary":
        value["counts"]["adapter_failure"] = 1
    assert buffer.append_agent_observation(row) == 0
    if damage == "independent_unknown":
        buffer.failures.add("capture_incomplete")
    document = buffer.document()
    if damage == "historical_generic":
        document["failures"] = ["capture_incomplete"]
    saved, raw = observations.build_snapshot("run", "execute", document,
        lease_generation=1, claim_request_hash=contracts.hash_bytes(b"claim"))

    class Objects:
        def get_bounded(self, ref, maximum):
            return raw

    readback = observations.read_snapshot(Objects(), "run", "execute", 1, contracts.hash_bytes(b"claim"))
    assert readback == saved
    example["worker_observation_artifacts"]["run"] = readback
    if damage == "legacy":
        example["configuration"].pop("training_maximum_missing_reasoning_percent")
    elif damage == "missing_host":
        readback["payload"]["response_receipts"].pop()
        readback["payload"]["response_receipt_count"] -= 1
    elif damage == "privacy":
        example["provider_calls"][0]["request_capture"]["redacted"] = True
    elif damage == "route":
        example["provider_calls"][0]["response"]["model"] = "unapproved/model"
    elif damage == "cost":
        example["provider_calls"][0]["actual_microusd"] = None
    elif damage == "output":
        example["output"]["changed"] = True
    elif damage == "input":
        example["input_snapshots"]["run"]["input"]["changed"] = True
    return refresh(example)


def assess(example):
    return training_eligibility.assess_capture(submission(), {"run": example}, ["run"], enabled=True)


def target_count(example):
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    return len(splits["train"]) + len(splits["eval"])


def test_actual_recorder_ack_transport_failure_keeps_exact_provider_capture(monkeypatch):
    fixture = captured_timeout()
    provider_call = fixture["provider_calls"][0]
    identity = provider_call["call_identity"]
    response_bytes = contracts.canonical_json(provider_call["response"]).encode("utf-8")
    receipt = fixture["worker_observation_artifacts"]["run"]["payload"]["response_receipts"][0]
    assert receipt["body_sha256"] == contracts.hash_bytes(response_bytes)
    assert receipt["body_bytes"] == len(response_bytes)
    with TemporaryDirectory(prefix="ack-", dir="/tmp") as directory:
        monkeypatch.setenv(checkpoint.WORKER_SOCKET_ENV, str(Path(directory) / "missing.sock"))
        recorder = automatic_capture.Recorder(checkpoint, memory_limit=1024, queue_limit=1, close_timeout=0.25)
        try:
            body = recorder._new("httpx", "https://openrouter.ai/api/v1/chat/completions", 200, {
                "content-type": "application/json", "x-leadpoet-call-identity": identity})
            assert body is not None and body.identity == identity
            body.add(response_bytes)
            body.finish(True)
            started = time.monotonic()
            summary = recorder.close()
            assert time.monotonic() - started < 0.5
        finally:
            recorder.close()
    assert not recorder.worker.is_alive()
    assert summary["complete"] is False
    assert summary["pending_queue"] == summary["inflight"] == 0
    assert summary["counts"] == {
        "responses_started": 1, "responses_complete": 1, "client_httpx": 1,
        "checkpoint_failure": 1, "checkpoint_failure_response_received_transport_error": 1}

    current_classifier = observations.optional_capture_failure

    def pre_fix_classifier(row):
        # Before this fix every response_received checkpoint failure was outside
        # the summary whitelist. Keep all other classification paths unchanged.
        value = row.get("data", {}).get("value", {})
        if any(key.startswith("checkpoint_failure_response_received_") for key in value.get("counts", {})):
            return False
        return current_classifier(row)

    with monkeypatch.context() as scope:
        scope.setattr(observations, "optional_capture_failure", pre_fix_classifier)
        previous = captured_failure(runtime_summary=summary)
        assert previous["worker_observation_artifacts"]["run"]["payload"]["failures"] == ["capture_incomplete"]
        assert all(training_steps._host_result(previous, call) for call in previous["provider_calls"])
        assert assess(previous)["eligible"]
        assert target_count(previous) == 2

    example = captured_failure(runtime_summary=summary)
    before = deepcopy(example)
    payload = example["worker_observation_artifacts"]["run"]["payload"]
    assert payload["complete"] is False
    assert payload["failures"] == [observations.OPTIONAL_CAPTURE_INCOMPLETE]
    assert payload["agent_observations"][0]["data"]["value"] == summary
    assert all(training_steps._host_result(example, call) for call in example["provider_calls"])
    assert assess(example)["eligible"]
    assert target_count(example) == 2
    assert example == before


@pytest.mark.parametrize("category", ["web_fetch", "response_received"])
@pytest.mark.parametrize("reason", ["timeout", "transport_error", "socket_unavailable"])
def test_known_checkpoint_failure_preserves_raw_snapshot_and_exact_provider_targets(category, reason):
    example = captured_failure(category=category, reason=reason)
    before = deepcopy(example)
    payload = example["worker_observation_artifacts"]["run"]["payload"]
    assert payload["complete"] is False
    assert payload["failures"] == [observations.OPTIONAL_CAPTURE_INCOMPLETE]
    assert all(training_steps._host_result(example, call) for call in example["provider_calls"])
    assert assess(example)["eligible"]
    assert target_count(example) == 2
    assert example == before


@pytest.mark.parametrize("reason", ["host_rejected", "invalid_response", "invalid_frame", "unknown"])
def test_rejected_or_invalid_client_ack_remains_generic(reason):
    example = captured_failure(reason=reason)
    assert example["worker_observation_artifacts"]["run"]["payload"]["failures"] == ["capture_incomplete"]
    assert assess(example)["eligible"]
    assert target_count(example) == 2


@pytest.mark.parametrize("damage", ["legacy", "independent_unknown", "historical_generic", "missing_host",
    "pending_queue", "inflight", "unbalanced", "unknown_summary", "privacy", "route", "cost", "input", "output"])
def test_failed_client_ack_cannot_waive_required_or_unknown_evidence(damage):
    example = captured_failure(damage=damage)
    before = deepcopy(example)
    if damage == "independent_unknown":
        assert example["worker_observation_artifacts"]["run"]["payload"]["failures"] == [
            "capture_incomplete", observations.OPTIONAL_CAPTURE_INCOMPLETE]
    diagnostic_only = damage in {"independent_unknown", "historical_generic", "pending_queue",
                                  "inflight", "unbalanced", "unknown_summary", "cost"}
    assert assess(example)["eligible"] is diagnostic_only
    expected = 2 if diagnostic_only or damage in {"privacy", "route"} else (1 if damage == "missing_host" else 0)
    assert target_count(example) == expected
    assert example == before


@pytest.mark.parametrize("failed_writes", [1, 18])
def test_client_ack_failure_with_failed_host_writes_never_produces_targets(failed_writes):
    example = captured_failure(failed_writes=failed_writes)
    payload = example["worker_observation_artifacts"]["run"]["payload"]
    assert sum(row["write_state"] == "socket_write_failed" for row in payload["response_receipts"]) == failed_writes
    assert payload["failures"] == ["capture_incomplete", observations.OPTIONAL_CAPTURE_INCOMPLETE]
    gate = training_eligibility.assess_capture(submission(), {"run": example}, ["run"], enabled=True,
                                             minimum_coverage_percent=80)
    assert not gate["eligible"]
    assert gate["counts"]["complete_runs"] == 0
    assert gate["counts"]["incomplete_runs"] == 1
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits['train'] + splits['eval']
    assert len(records) == 1
    failed = {row['call_identity'] for row in payload['response_receipts'] if row['write_state'] == 'socket_write_failed'}
    assert all(ref['call_identity'] not in failed for row in records for ref in row['source_refs'])
