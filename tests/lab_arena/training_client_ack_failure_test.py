"""Optional read-report failures cannot invalidate exact host-delivered results."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_observations as observations, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import submission
from tests.lab_arena.training_optional_capture_test import captured_timeout, refresh
from tests.lab_arena.training_worker_proof_reuse_test import large_example


def captured_failure(*, category="response_received", reason="transport_error", damage=None,
                     failed_writes=0):
    template = captured_timeout()
    example = large_example(20) if failed_writes else deepcopy(template)
    if failed_writes:
        for key in ("completeness", "attempts", "related_run_gap_attribution"):
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
    value = row["data"]["value"]
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
    assert not assess(example)["eligible"]
    assert target_count(example) == 0


@pytest.mark.parametrize("damage", ["legacy", "independent_unknown", "historical_generic", "missing_host",
    "pending_queue", "inflight", "unbalanced", "unknown_summary", "privacy", "route", "cost", "input", "output"])
def test_failed_client_ack_cannot_waive_required_or_unknown_evidence(damage):
    example = captured_failure(damage=damage)
    before = deepcopy(example)
    if damage == "independent_unknown":
        assert example["worker_observation_artifacts"]["run"]["payload"]["failures"] == [
            "capture_incomplete", observations.OPTIONAL_CAPTURE_INCOMPLETE]
    assert not assess(example)["eligible"]
    assert target_count(example) == 0
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
    assert target_count(example) == 0
