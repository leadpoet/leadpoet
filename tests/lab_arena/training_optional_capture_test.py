"""Optional HTTP diagnostics never replace proof of required provider bytes."""
from copy import deepcopy
import base64

import pytest

from lab_arena import contracts, training_eligibility, training_observations as observations, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import submission
from tests.lab_arena.training_host_result_test import local_runtime_diagnostics
from tests.lab_arena.training_steps_test import seal


OPTIONAL = "optional_capture_incomplete"


def summary(**extra_counts):
    return {"id": 0, "kind": "state_snapshot", "observed_at": "2026-10-25T00:00:00Z",
        "after_action_sequence": 2, "depends_on": [],
        "privacy": {"redacted": False, "redactions": []},
        "data": {"name": "automatic_capture_summary", "value": {
            "source": "runtime_http_v1", "complete": False, "pending_queue": 0, "inflight": 0,
            "counts": {"responses_started": 1, "responses_complete": 1, "client_httpx": 1,
                       "checkpoint_failure": 1, "checkpoint_failure_web_fetch_timeout": 1,
                       **extra_counts}}}}


def refresh(example):
    artifact = example["worker_observation_artifacts"]["run"]
    artifact["payload_sha256"] = contracts.document_hash(artifact["payload"])
    for call in example["provider_calls"]:
        if call.get("run_id") == "run":
            call["response_provenance"]["worker_receipt_ref"]["payload_sha256"] = artifact["payload_sha256"]
    return seal(example)


def captured_timeout():
    example = local_runtime_diagnostics(redacted=False)
    buffer = observations.Buffer()
    for call, receipt in zip(example["provider_calls"], example["worker_observation_artifacts"]["run"]["payload"]["response_receipts"]):
        call["call_identity"] = contracts.hash_bytes(call["call_identity"].encode())
        receipt = {**receipt, "call_identity": call["call_identity"], "protocol": "frame",
                   "socket_written_at": "2026-10-25T00:00:00Z"}
        assert buffer.append_response_receipt(receipt)
    assert buffer.append_agent_observation(summary()) == 0
    snapshot, raw = observations.build_snapshot("run", "execute", buffer.document(),
        lease_generation=1, claim_request_hash=contracts.hash_bytes(b"claim"))
    class Objects:
        def get_bounded(self, ref, maximum):
            return raw
    saved = observations.read_snapshot(Objects(), "run", "execute", 1, contracts.hash_bytes(b"claim"))
    assert saved == snapshot
    example["worker_observation_artifacts"]["run"] = saved
    dimensions = {name: [] for name in ("context", "source_identity", "exact_inputs", "input_output_label", "costs", "research_process")}
    dimensions["research_process"] = ["worker_observations_incomplete"]
    example["completeness"] = {"complete": False, "incomplete_reasons": ["worker_observations_incomplete"],
        "dimensions": {name: not values for name, values in dimensions.items()}, "reasons_by_dimension": dimensions}
    example["attempts"] = [{"run_id": "run", "kind": "execute", "status": "accepted"},
                           {"run_id": "score", "kind": "score", "status": "accepted"}]
    example["related_run_gap_attribution"] = {
        "run": {"kind": "execute", "status": "accepted", "research_process_gaps": ["worker_observations_incomplete"], "cost_gaps": []},
        "score": {"kind": "score", "status": "accepted", "research_process_gaps": [], "cost_gaps": []}}
    return refresh(example)


def redacted_parse_row(ordinal):
    """A rejected local HTTP request with no admitted provider identity."""
    return {"ordinal": ordinal, "action_sequence": None, "operation_id": "unknown",
        "kind": "local_parse_error", "call_identity": None,
        "contributes_to_provider_call_count": False,
        "request": {"admitted_parameters": {"method": "POST",
            "url": "https://openrouter.ai/api/v1/responses",
            "headers": {"authorization": "Bearer synthetic-private-value"},
            "body_text": {"model": "synthetic/model"}}, "timeout_ms": 0},
        "outcome": {"protocol": "http", "error_code": "invalid_request",
            "response": {"status": 400, "headers": {}, "body_b64": base64.b64encode(
                b'{"error":{"code":"invalid_request"}}').decode()}},
        "privacy": {"redacted": False, "redactions": []}}


@pytest.mark.parametrize("parse_count,receipt_count", [(4, 467), (2, 422)])  # Saved ICP4/7 shapes.
def test_optional_capture_accepts_only_redacted_noncontributing_parse_diagnostics(parse_count, receipt_count):
    buffer = observations.Buffer()
    assert buffer.append_agent_observation(summary()) == 0
    for ordinal in range(receipt_count):
        assert buffer.append_response_receipt({
            "ordinal": ordinal, "action_sequence": ordinal, "operation_id": "openrouter.chat",
            "call_identity": contracts.hash_bytes(str(ordinal).encode()), "protocol": "http",
            "status": 200, "body_sha256": contracts.hash_bytes(b"synthetic"), "body_bytes": 9,
            "write_state": "socket_written", "socket_written_at": "2026-10-25T00:00:00Z"})
    for ordinal in range(parse_count):
        buffer.append(redacted_parse_row(receipt_count + ordinal))
    payload = buffer.document()
    assert payload["failures"] == [OPTIONAL]
    assert payload["observation_count"] == parse_count
    assert payload["response_receipt_count"] == receipt_count
    assert all(row["privacy"] == {"redacted": True, "redactions": ["credential"]}
               for row in payload["observations"])
    assert observations.optional_capture_only(payload)
    payload["observations"][0]["kind"] = "transport_failure"
    payload["observations"][0]["operation_id"] = "openrouter.chat"
    payload["observations"][0]["action_sequence"] = 0
    payload["observations"][0]["contributes_to_provider_call_count"] = True
    assert not observations.optional_capture_only(payload)


@pytest.mark.parametrize("mutation", ["hidden_reasoning", "call_identity", "action_sequence",
                                     "provider_count", "timeout", "error_code", "response"])
def test_local_parse_diagnostic_exception_requires_exact_rejected_shape(mutation):
    buffer = observations.Buffer()
    assert buffer.append_agent_observation(summary()) == 0
    buffer.append(redacted_parse_row(0))
    payload = buffer.document()
    row = payload["observations"][0]
    if mutation == "hidden_reasoning":
        row["privacy"]["redactions"].append("hidden_reasoning")
    elif mutation == "call_identity":
        row["call_identity"] = contracts.hash_bytes(b"admitted")
    elif mutation == "action_sequence":
        row["action_sequence"] = 0
    elif mutation == "provider_count":
        row["contributes_to_provider_call_count"] = True
    elif mutation == "timeout":
        row["request"]["timeout_ms"] = 1000
    elif mutation == "error_code":
        row["outcome"].pop("error_code")
    else:
        row["outcome"].pop("response")
    assert not observations.optional_local_parse_redaction(row)
    assert not observations.optional_capture_only(payload)


def test_native_export_keeps_parse_privacy_but_selects_exact_provider_view():
    from lab_arena import training_export
    from tests.lab_arena.training_export_receipts_test import values_with_receipts
    from tests.lab_arena.training_observations_test import save
    from tests.lab_arena.training_eligibility_test import captured_example

    values, original = values_with_receipts()
    values["round_row"]["configuration_doc"].update(
        captured_example()["configuration"], training_maximum_missing_reasoning_percent=0)
    buffer = observations.Buffer()
    for receipt in original["response_receipts"]:
        assert buffer.append_response_receipt(receipt)
    for row in original["agent_observations"]:
        assert buffer.append_agent_observation(row) is not None
    assert buffer.append_agent_observation({**summary(), "id": len(buffer.agent_observations)}) is not None
    for ordinal in range(4):
        buffer.append(redacted_parse_row(ordinal + 1))
    saved = save(values, buffer.document())
    example = training_export.assemble_example(**values)
    assert example["worker_observation_artifacts"]["execute:1"] == saved
    assert set(example["completeness"]["incomplete_reasons"]) == {
        "worker_observations_incomplete", "worker_observation_privacy_nonexact"}
    assert example["privacy"]["privacy_filtered"] is True
    assert example["privacy"]["exact_replay"] is False
    view, ignored = training_thinking.provider_visible_view(example)
    assert ignored == {"worker_observations_incomplete", "worker_observation_privacy_nonexact"}
    assert view["completeness"]["complete"] is True
    assert example["completeness"]["complete"] is False
    assert example["worker_observation_artifacts"]["execute:1"]["payload"]["observations"] == saved["payload"]["observations"]


@pytest.mark.parametrize("damage", ["generic_failure", "failed_write"])
def test_parse_diagnostic_never_covers_required_capture_damage(damage):
    example = captured_timeout()
    payload = example["worker_observation_artifacts"]["run"]["payload"]
    payload["observations"] = [observations.clean_observation(redacted_parse_row(10))]
    payload["observation_count"] = 1
    if damage == "generic_failure":
        payload["failures"].insert(0, "capture_incomplete")
    else:
        payload["response_receipts"][0].update(write_state="socket_write_failed", socket_written_at=None)
    for values in (example["completeness"]["incomplete_reasons"],
                   example["completeness"]["reasons_by_dimension"]["research_process"],
                   example["related_run_gap_attribution"]["run"]["research_process_gaps"]):
        values.append("worker_observation_privacy_nonexact")
    refresh(example)
    assert not observations.optional_capture_only(payload)
    view, ignored = training_thinking.provider_visible_view(example)
    assert view is example and not ignored


def test_mixed_related_exact_score_artifact_preserves_optional_source_view():
    example = captured_timeout()
    payload = example["worker_observation_artifacts"]["run"]["payload"]
    payload["observations"] = [observations.clean_observation(redacted_parse_row(10))]
    payload["observation_count"] = 1
    score, _ = observations.build_snapshot("score", "score", observations.Buffer().document(),
        lease_generation=1, claim_request_hash=contracts.hash_bytes(b"score-claim"))
    example["worker_observation_artifacts"]["score"] = score
    for values in (example["completeness"]["incomplete_reasons"],
                   example["completeness"]["reasons_by_dimension"]["research_process"],
                   example["related_run_gap_attribution"]["run"]["research_process_gaps"]):
        values.append("worker_observation_privacy_nonexact")
    refresh(example)
    view, ignored = training_thinking.provider_visible_view(example)
    assert ignored == {"worker_observations_incomplete", "worker_observation_privacy_nonexact"}
    assert view["completeness"]["complete"] is True
    gate = training_eligibility.assess_capture(submission(), {"run": example}, ["run"], enabled=True)
    assert gate["eligible"], gate
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    assert len(splits["train"]) + len(splits["eval"]) == 2


def test_mixed_related_required_redaction_stays_hard():
    from tests.lab_arena.training_observations_test import observation

    example = captured_timeout()
    source = example["worker_observation_artifacts"]["run"]["payload"]
    source["observations"] = [observations.clean_observation(redacted_parse_row(10))]
    source["observation_count"] = 1
    score_buffer = observations.Buffer()
    required = observation(kind="transport_failure", action=0, ordinal=0, contributes=True)
    required["request"]["admitted_parameters"]["authorization"] = "Bearer synthetic-private-value"
    score_buffer.append(required)
    score, _ = observations.build_snapshot("score", "score", score_buffer.document(),
        lease_generation=1, claim_request_hash=contracts.hash_bytes(b"score-claim"))
    example["worker_observation_artifacts"]["score"] = score
    for values in (example["completeness"]["incomplete_reasons"],
                   example["completeness"]["reasons_by_dimension"]["research_process"],
                   example["related_run_gap_attribution"]["run"]["research_process_gaps"],
                   example["related_run_gap_attribution"]["score"]["research_process_gaps"]):
        values.append("worker_observation_privacy_nonexact")
    refresh(example)
    view, ignored = training_thinking.provider_visible_view(example)
    assert view is example and not ignored


def test_timeout_source_stored_artifact_and_provider_dataset_agree():
    example = captured_timeout()
    before = deepcopy(example)
    payload = example["worker_observation_artifacts"]["run"]["payload"]
    assert payload["complete"] is False and payload["failures"] == [OPTIONAL]
    assert all(training_steps._host_result(example, call) for call in example["provider_calls"])
    gate = training_eligibility.assess_capture(submission(), {"run": example}, ["run"], enabled=True)
    assert gate["eligible"], gate
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits["train"] + splits["eval"]
    assert len(records) == 2
    assert all(record["prior_observations"] == [] for record in records)
    assert example == before


@pytest.mark.parametrize("counts", [
    {"queue_overflow": 1}, {"provider_ack_ineligible": 1}, {"adapter_failure": 1},
    {"checkpoint_failure_response_received_timeout": 1}, {"checkpoint_failure": 2},
    {"checkpoint_failure": True}, {"checkpoint_failure_web_fetch_timeout": -1},
    {"responses_started": 2}, {"web_written": 1},
])
def test_unattributed_or_combined_summary_failure_remains_generic(counts):
    buffer = observations.Buffer()
    assert buffer.append_agent_observation(summary(**counts)) == 0
    assert buffer.document()["failures"] == ["capture_incomplete"]


def test_independent_unknown_failure_remains_distinct_at_source():
    buffer = observations.Buffer()
    assert buffer.append_agent_observation(summary()) == 0
    buffer.failures.add("capture_incomplete")  # Existing host/control failure path.
    assert buffer.document()["failures"] == ["capture_incomplete", OPTIONAL]
    assert not observations.optional_capture_only(buffer.document())


@pytest.mark.parametrize("failure", [{}, [], "unknown_failure"])
def test_failure_enum_rejects_untrusted_values_with_validation_error(failure):
    buffer = observations.Buffer()
    assert buffer.append_agent_observation(summary()) == 0
    payload = buffer.document()
    payload["failures"] = [failure]
    with pytest.raises(observations.ObservationError, match="counts_invalid"):
        observations._validate_payload(payload)


@pytest.mark.parametrize("new_policy", [False, True])
@pytest.mark.parametrize("checkpoint_category", ["web_fetch", "response_received"])
def test_actual_export_preserves_optional_source_tag_and_raw_incompleteness(new_policy, checkpoint_category):
    from lab_arena import training_export
    from tests.lab_arena.training_export_receipts_test import values_with_receipts
    from tests.lab_arena.training_observations_test import save
    from tests.lab_arena.training_eligibility_test import captured_example
    values, payload = values_with_receipts()
    config = values["round_row"]["configuration_doc"]
    config.update(captured_example()["configuration"], training_maximum_missing_reasoning_percent=0)
    if not new_policy:
        config.pop("training_maximum_missing_reasoning_percent")
    buffer = observations.Buffer()
    assert buffer.append_response_receipt(payload["response_receipts"][0])
    runtime_summary = summary()
    if checkpoint_category == "response_received":
        runtime_summary["data"]["value"]["counts"] = {
            "responses_started": 1, "responses_complete": 1, "client_httpx": 1,
            "checkpoint_failure": 1, "checkpoint_failure_response_received_transport_error": 1}
    assert buffer.append_agent_observation(runtime_summary) == 0
    saved = save(values, buffer.document())
    example = training_export.assemble_example(**values)
    artifact = example["worker_observation_artifacts"]["execute:1"]
    assert artifact == saved
    assert artifact["payload"]["complete"] is False
    assert artifact["payload"]["failures"] == [OPTIONAL]
    view, diagnostics = training_thinking.provider_visible_view(example)
    if new_policy:
        assert example["completeness"]["incomplete_reasons"] == ["worker_observations_incomplete"]
        assert diagnostics == {"worker_observations_incomplete"}
        assert view["completeness"]["complete"] is True
    else:
        assert not diagnostics and view is example
        assert set(example["completeness"]["incomplete_reasons"]) == {
            "worker_observations_incomplete", "agent_response_receipts_unverified", "agent_task_state_unverified"}
    assert example["completeness"]["complete"] is False


@pytest.mark.parametrize("cost_gap", [False, True])
def test_optional_timeout_composes_with_independent_judge_gap(cost_gap):
    from tests.lab_arena.training_eligibility_test import judge_only_gap_example
    example = captured_timeout()
    judge = judge_only_gap_example(cost_gap=cost_gap)
    example["provider_calls"].append(judge["provider_calls"][-1])
    gaps = judge["completeness"]["incomplete_reasons"]
    proof = example["completeness"]
    proof["incomplete_reasons"] = sorted(set(proof["incomplete_reasons"]) | set(gaps))
    proof["reasons_by_dimension"]["research_process"] = proof["incomplete_reasons"][:]
    proof["reasons_by_dimension"]["costs"] = ["provider_cost_unresolved"] if cost_gap else []
    proof["dimensions"]["costs"] = not cost_gap
    example["related_run_gap_attribution"]["score"].update(
        research_process_gaps=gaps[:], cost_gaps=proof["reasons_by_dimension"]["costs"][:])
    seal(example)
    before = deepcopy(example)
    gate = training_eligibility.assess_capture(submission(), {"run": example}, ["run"], enabled=True)
    assert gate["eligible"], gate
    splits, _ = training_thinking.build([example])
    assert len(splits["train"]) + len(splits["eval"]) == 2
    assert example == before


@pytest.mark.parametrize("damage", ["unknown", "old_generic", "missing_host", "failed_write", "host_hash",
    "receipt_count", "observation_count", "agent_count", "privacy", "payload_hash", "partial",
    "missing_snapshot", "contradictory_ack", "legacy", "another_unknown_run", "unattributed_unknown_artifact",
    "partial_journal", "duplicate_host", "extra_host", "duplicate_call", "missing_attribution", "missing_attempt",
    "run_identity", "kind", "claim", "generation", "summary_queue", "unknown_tag"])
def test_optional_flag_does_not_waive_required_or_unknown_failures(damage):
    example = captured_timeout()
    artifact = example["worker_observation_artifacts"]["run"]
    payload = artifact["payload"]
    if damage == "unknown": payload["failures"].append("capture_incomplete")
    elif damage == "old_generic": payload["failures"] = ["capture_incomplete"]
    elif damage == "missing_host":
        payload["response_receipts"].pop()
        payload["response_receipt_count"] -= 1
        assert observations.optional_capture_only(payload)
    elif damage == "failed_write":
        payload["response_receipts"][0].update(write_state="socket_write_failed", socket_written_at=None)
        assert observations._validate_payload(payload) == payload
    elif damage == "host_hash": payload["response_receipts"][0]["body_sha256"] = contracts.hash_bytes(b"different")
    elif damage == "receipt_count": payload["response_receipt_count"] += 1
    elif damage == "observation_count": payload["observation_count"] += 1
    elif damage == "agent_count": payload["agent_observation_count"] += 1
    elif damage == "privacy":
        from tests.lab_arena.training_observations_test import observation
        row = observation()
        row["privacy"] = {"redacted": True, "redactions": ["credential"]}
        payload.update(observations=[row], observation_count=1)
        assert observations._validate_payload(payload) == payload
    elif damage == "partial": artifact["complete"] = False
    elif damage == "contradictory_ack": example["provider_calls"][0]["response_provenance"]["agent_receipt"] = "reported_exact_body_received"
    elif damage == "legacy": example["configuration"].pop("training_maximum_missing_reasoning_percent")
    elif damage == "another_unknown_run":
        example["related_run_gap_attribution"]["score"]["research_process_gaps"] = ["worker_observations_incomplete"]
    elif damage == "unattributed_unknown_artifact":
        bad = deepcopy(artifact)
        bad.update(run_id="score", kind="score")
        bad["payload"]["failures"] = ["capture_incomplete"]
        bad["payload_sha256"] = contracts.document_hash(bad["payload"])
        example["worker_observation_artifacts"]["score"] = bad
    elif damage == "partial_journal": example["partial_worker_observation_artifacts"] = {"score": {"complete": False}}
    elif damage in {"duplicate_host", "extra_host"}:
        receipt = deepcopy(payload["response_receipts"][-1])
        receipt["ordinal"] += 1
        if damage == "extra_host": receipt["call_identity"] = contracts.hash_bytes(b"unlinked")
        payload["response_receipts"].append(receipt)
        payload["response_receipt_count"] += 1
    elif damage == "duplicate_call": example["provider_calls"].append(deepcopy(example["provider_calls"][0]))
    elif damage == "missing_attribution": example.pop("related_run_gap_attribution")
    elif damage == "missing_attempt": example["attempts"].pop()
    elif damage == "run_identity": artifact["run_id"] = "other"
    elif damage == "kind": artifact["kind"] = "score"
    elif damage == "claim": artifact["claim_request_hash"] = "invalid"
    elif damage == "generation": artifact["lease_generation"] = True
    elif damage == "summary_queue": payload["agent_observations"][0]["data"]["value"]["counts"]["queue_overflow"] = 1
    elif damage == "unknown_tag": payload["failures"].append("unknown_failure")
    refresh(example)
    if damage == "payload_hash": artifact["payload_sha256"] = contracts.hash_bytes(b"wrong")
    elif damage == "missing_snapshot": example["worker_observation_artifacts"] = {}
    seal(example)
    required_failure = damage in {"missing_host", "failed_write", "host_hash", "payload_hash",
                                  "missing_snapshot", "contradictory_ack", "legacy", "missing_attempt",
                                  "run_identity", "kind", "claim", "generation"}
    result = training_eligibility.assess_capture(submission(), {"run": example}, ["run"], enabled=True)
    assert result["eligible"] is (not required_failure), result
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits["train"] + splits["eval"]
    expected = (0 if damage in {"payload_hash", "missing_snapshot", "legacy", "missing_attempt",
                                 "run_identity", "kind", "claim", "generation"}
                else 1 if damage == "missing_host" else 2)
    assert len(records) == expected
