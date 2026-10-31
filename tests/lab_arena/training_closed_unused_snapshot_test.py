"""Closed unused requests and matching immutable host capture keep their proof."""

from copy import deepcopy
import gzip

import pytest

from lab_arena import contracts, training_export, training_observations, training_steps, training_thinking
from tests.lab_arena.training_optional_capture_test import summary
from tests.lab_arena.training_retry_snapshot_test import retry_values, save_snapshot
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_undispatched_coverage_test import assess_bank, undispatched_values


def closed_unused_values(operation="openrouter.responses"):
    values = undispatched_values(operation=operation, complete=True)
    run = values["runs"][0]
    payload = training_observations.read_snapshot(values["objects"], run["run_id"], "execute",
        run["lease_generation"], run["claim_request_hash"])["payload"]
    payload.update(observations=[], observation_count=0)
    save_snapshot(values, run, payload)
    return values


def strict_pairs(example):
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    return [(row["input"], row["target"]) for split in ("train", "eval") for row in splits[split]]


@pytest.mark.parametrize("operation", ["deepline.execute", "openrouter.chat", "openrouter.responses"])
def test_closed_unused_reservation_needs_no_worker_exception_or_model_target(operation):
    values = closed_unused_values(operation)
    before_runs, before_ledger = deepcopy(values["runs"]), deepcopy(values["ledger_rows"])
    example = training_export.assemble_example(**values)
    call = example["provider_calls"][-1]
    assert "worker_observation_ref" not in call
    assert training_thinking.verified_undispatched_transport(example, call)
    result = assess_bank(values, example)
    assert result["eligible"], result
    assert result["counts"]["complete_runs"] == 10
    assert result["counts"]["incomplete_runs"] == 0
    assert "response" not in call and call["actual_microusd"] == 0
    baseline = deepcopy(example)
    baseline["provider_calls"].pop()
    seal(baseline)
    assert len(strict_pairs(example)) == 2
    assert strict_pairs(example) == strict_pairs(baseline)
    with pytest.raises(ValueError):
        training_steps._candidate(example, call)
    assert values["runs"] == before_runs and values["ledger_rows"] == before_ledger


@pytest.mark.parametrize("damage", ["dispatch", "reason", "amount", "reserved_amount", "request",
    "run", "response", "upstream_receipt", "host_write", "failed_host_write", "agent_receipt",
    "sequence_collision", "identity_elsewhere", "observation_ref"])
def test_closed_unused_proof_rejects_every_contradiction(damage):
    example = training_export.assemble_example(**closed_unused_values())
    call = example["provider_calls"][-1]
    artifact = example["worker_observation_artifacts"][example["run_id"]]
    payload = artifact["payload"]
    if damage == "dispatch":
        call["lifecycle"].insert(1, {**deepcopy(call["lifecycle"][0]), "entry_kind": "dispatch"})
    elif damage == "reason": call["lifecycle"][-1]["accounting"]["reason"] = "other"
    elif damage == "amount": call["lifecycle"][-1]["amount_microusd"] = 1
    elif damage == "reserved_amount": call["lifecycle"][-1]["accounting"]["reserved_microusd"] = 1
    elif damage == "request": call["request"]["request_hash"] = contracts.hash_bytes(b"wrong")
    elif damage == "run": call["run_id"] = "another-run"
    elif damage == "response": call["response"] = {}
    elif damage == "upstream_receipt": call["upstream_response_receipts"] = [{"status": 200}]
    elif damage in {"host_write", "failed_host_write"}:
        row = deepcopy(payload["response_receipts"][0])
        row.update(ordinal=9, action_sequence=call["action_sequence"], call_identity=call["call_identity"],
                   operation_id=call["operation_id"])
        if damage == "failed_host_write":
            row.update(write_state="socket_write_failed", socket_written_at=None)
            payload.update(complete=False, failures=["capture_incomplete"])
        payload["response_receipts"].append(row)
        payload["response_receipt_count"] += 1
    elif damage == "agent_receipt":
        row = payload["response_receipts"][0]
        payload["agent_observations"].append({"id": 0, "kind": "response_received", "observed_at": "2026-10-28T00:00:01Z",
            "after_action_sequence": call["action_sequence"], "depends_on": [],
            "privacy": {"redacted": False, "redactions": []}, "data": {
                **{key: row[key] for key in ("status", "body_sha256", "body_bytes")},
                "call_identity": call["call_identity"]}})
        payload["agent_observation_count"] += 1
    elif damage in {"sequence_collision", "identity_elsewhere"}:
        source = training_export.assemble_example(**undispatched_values(complete=True))
        row = deepcopy(source["worker_observation_artifacts"][source["run_id"]]["payload"]["observations"][0])
        if damage == "identity_elsewhere": row.update(action_sequence=99, call_identity=call["call_identity"])
        payload["observations"].append(row)
        payload["observation_count"] = 1
    else: call["worker_observation_ref"] = {"run_id": call["run_id"], "ordinal": 0}
    assert training_observations._validate_payload(payload) == payload
    artifact["payload_sha256"] = contracts.document_hash(payload)
    seal(example)
    assert not training_thinking.verified_undispatched_transport(example, call)


def test_other_managed_provider_attempt_can_share_action_sequence_and_write():
    values = closed_unused_values("openrouter.chat")
    # The broker deliberately reuses one action sequence across managed
    # provider attempts. Another attempt's host write does not dispatch this
    # independently closed reservation.
    reservation = values["ledger_rows"][-2]
    reservation["entry_doc"]["action_sequence"] = 1
    reservation["entry_doc"]["provider_attempt"] = 2
    values["runs"][0]["result_doc"]["resource_summary"]["provider_call_count"] -= 1
    example = training_export.assemble_example(**values)
    call = next(call for call in example["provider_calls"] if call["accounting_state"] == "recovery")
    receipts = example["worker_observation_artifacts"][example["run_id"]]["payload"]["response_receipts"]
    assert any(row["action_sequence"] == call["action_sequence"]
               and row["call_identity"] != call["call_identity"] for row in receipts)
    assert training_thinking.verified_undispatched_transport(example, call)
    assert assess_bank(values, example)["counts"]["complete_runs"] == 10
    assert len(strict_pairs(example)) == 2


def unconfirmed_native_values(*, complete=False, supplied_hash=False):
    values, predecessor, _ = retry_values()
    values["runs"].remove(predecessor)
    run = values["runs"][0]
    payload = training_observations.read_snapshot(values["objects"], run["run_id"], "execute",
        run["lease_generation"], run["claim_request_hash"])["payload"]
    if not complete:
        buffer = training_observations.Buffer()
        row = summary()
        row["data"]["value"]["counts"] = {"checkpoint_failure": 1,
            "checkpoint_failure_state_transport_error": 1}
        buffer.append_agent_observation(row)
        payload.update(agent_observations=buffer.document()["agent_observations"],
            agent_observation_count=1, complete=False, failures=buffer.document()["failures"])
    snapshot = save_snapshot(values, run, payload)
    capture = {"status": "unavailable", **{key: payload[key] for key in (
        "observation_count", "local_refusal_count", "complete")}}
    if supplied_hash:
        capture["payload_sha256"] = snapshot["payload_sha256"]
    run["result_doc"]["resource_summary"]["worker_observations"] = capture
    return values, run, snapshot


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("supplied_hash", [False, True])
def test_unconfirmed_native_snapshot_proves_host_bytes_without_rewriting_summary(complete, supplied_hash):
    values, run, snapshot = unconfirmed_native_values(complete=complete, supplied_hash=supplied_hash)
    before = deepcopy(run)
    exported = training_export.assemble_example(**values)
    assert exported["worker_observation_artifacts"][run["run_id"]] == snapshot
    assert not exported.get("unverified_worker_observation_artifacts")
    assert "worker_observation_summary_invalid" not in exported["completeness"]["incomplete_reasons"]
    assert ("worker_observations_incomplete" in exported["completeness"]["incomplete_reasons"]) is not complete
    assert snapshot["payload_sha256"] == contracts.document_hash(snapshot["payload"])
    for call in exported["provider_calls"]:
        training_steps._host_result(exported, call)
    result = assess_bank(values, exported)
    assert result["counts"]["complete_runs"] == 10, result
    baseline = deepcopy(values)
    baseline["runs"][0]["result_doc"]["resource_summary"].pop("worker_observations")
    assert strict_pairs(exported) == strict_pairs(training_export.assemble_example(**baseline))
    assert len(strict_pairs(exported)) == 2
    assert run == before


@pytest.mark.parametrize("damage", ["hash", "count", "count_type", "refusals", "complete", "complete_type",
    "missing_count", "status", "legacy", "lease", "claim", "run", "kind", "corrupt", "missing",
    "later_attempt", "failed_write", "unknown_identity_claim", "unknown_hash_claim"])
def test_unconfirmed_native_snapshot_does_not_waive_any_supplied_claim_or_host_proof(damage):
    values, run, snapshot = unconfirmed_native_values()
    capture = run["result_doc"]["resource_summary"]["worker_observations"]
    ref = training_observations.object_ref(run["run_id"], run["lease_generation"], run["claim_request_hash"])
    if damage == "hash": capture["payload_sha256"] = contracts.hash_bytes(b"wrong")
    elif damage == "count": capture["observation_count"] += 1
    elif damage == "count_type": capture["observation_count"] = False
    elif damage == "refusals": capture["local_refusal_count"] += 1
    elif damage == "complete": capture["complete"] = True
    elif damage == "complete_type": capture["complete"] = 0
    elif damage == "missing_count": capture.pop("observation_count")
    elif damage == "status": capture["status"] = "rejected"
    elif damage == "legacy": values["round_row"]["configuration_doc"].pop("training_maximum_missing_reasoning_percent")
    elif damage == "unknown_identity_claim": capture["run_id"] = "another run"
    elif damage == "unknown_hash_claim": capture["sha256"] = contracts.hash_bytes(b"wrong")
    elif damage in {"lease", "claim", "run", "kind"}:
        key = {"lease": "lease_generation", "claim": "claim_request_hash", "run": "run_id", "kind": "kind"}[damage]
        snapshot[key] = {"lease": 2, "claim": contracts.hash_bytes(b"wrong"), "run": "other", "kind": "score"}[damage]
        values["objects"].put(ref, gzip.compress(contracts.canonical_json(snapshot).encode(), mtime=0))
    elif damage == "corrupt": values["objects"].put(ref, b"not gzip")
    elif damage in {"missing", "later_attempt"}:
        del values["objects"].values[ref]
        if damage == "later_attempt":
            later = {**run, "lease_generation": run["lease_generation"] + 1}
            save_snapshot(values, later, snapshot["payload"])
    else:
        snapshot["payload"]["response_receipts"][0].update(write_state="socket_write_failed", socket_written_at=None)
        save_snapshot(values, run, snapshot["payload"])
    exported = training_export.assemble_example(**values)
    if damage == "failed_write":
        assert exported["provider_calls"][0]["response_provenance"]["delivery_to_worker"] != "socket_written"
    else:
        assert run["run_id"] not in exported["worker_observation_artifacts"]
    result = assess_bank(values, exported)
    assert result["counts"]["complete_runs"] == 9, result
