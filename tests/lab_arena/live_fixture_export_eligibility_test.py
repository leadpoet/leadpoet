"""Replay saved live provider bytes through the real export and capture gate.

Provider ledger rows, request objects, worker socket receipts and agent read
acknowledgements come from the bounded October 24 injected fixture run. Source
bytes match the reviewed fixture archive. Benchmark, input, output, judgment
and phase declarations are explicitly synthetic. This is not an admitted or
scored Arena submission.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

from lab_arena import contracts, scoring, source_bundle, training_eligibility, training_observations
from lab_arena.training_export import assemble_example
from tests.lab_arena.training_export_test import OUTPUT_REF, fixture, save_input


LIVE = Path(__file__).resolve().parents[2] / "tests/fixtures/research_review/live_replay"


def _context(*, add_phase_states):
    live = json.loads((LIVE / "ledger-readback.json").read_text())
    diagnostic = json.loads((LIVE / "diagnostic-context.json").read_text())
    rows = live["rows"]
    first = rows[0]
    run_id, round_id, submission_id = first["run_id"], first["round_id"], first["submission_id"]
    values = fixture()
    objects = values["objects"]
    round_row = values["round_row"]
    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "reviewed-fixture.tar.gz"
        source_bundle.write_source_archive(LIVE.parent / "adaptive_json", source_path)
        source = source_path.read_bytes()
    assert contracts.hash_bytes(source) == diagnostic["source_archive_sha256"]
    source_ref = f"arena/{round_id}/sources/{submission_id}.tar.gz"
    objects.put(source_ref, source)
    round_row.update(round_id=round_id, status="open", benchmark_ref="synthetic-benchmark",
                     evaluation_date="2099-10-14")
    round_row["configuration_doc"].update(diagnostic["configuration"])
    round_row["participants"] = [{"submission_id": submission_id,
                                  "source_ref": source_ref, "source_size_bytes": len(source)}]
    original_benchmark = json.loads(objects.values["benchmark"])
    original_benchmark["round_id"] = round_id
    objects.put("synthetic-benchmark", contracts.canonical_json(original_benchmark).encode())
    submission = values["submission_row"]
    submission.update(submission_id=submission_id, round_id=round_id,
                      miner_hotkey=first["miner_hotkey"], source_ref=source_ref,
                      source_size_bytes=len(source), source_archive_sha256=contracts.hash_bytes(source))
    execution, judge = values["runs"]
    execution.update(run_id=run_id, round_id=round_id, submission_id=submission_id,
                     miner_hotkey=first["miner_hotkey"], stage=first["stage"],
                     lease_generation=diagnostic["lease_generation"],
                     claim_request_hash=diagnostic["claim_request_hash"],
                     output_ref=f"arena/{round_id}/outputs/{run_id}.json")
    execution["result_doc"]["resource_summary"]["provider_call_count"] = diagnostic["model_calls"]
    objects.put(execution["output_ref"], objects.values[OUTPUT_REF])
    judge.update(run_id="synthetic-judge", round_id=round_id, submission_id=submission_id,
                 miner_hotkey=first["miner_hotkey"], stage=first["stage"],
                 scored_run_id=run_id, claim_request_hash=diagnostic["claim_request_hash"])
    old_judgment = json.loads(objects.values["judge-output"])
    objects.put("judge-output", contracts.canonical_json(scoring.build_scoring_output(
        run_id, old_judgment["breakdowns"])).encode())
    icp = original_benchmark["icps"][0]
    output = json.loads(objects.values[execution["output_ref"]])
    save_input(objects, run_id, "execute", {
        "schema_version": "leadpoet.lab_arena.icp_input.v1", "icp": icp,
        "evaluation_date": round_row["evaluation_date"]},
        generation=execution["lease_generation"], claim_hash=execution["claim_request_hash"])
    save_input(objects, judge["run_id"], "score", scoring.build_scoring_input(
        scored_run_id=run_id, icp=icp, companies=output["companies"],
        policy=round_row["configuration_doc"]["scorer_policy"],
        evaluation_date=round_row["evaluation_date"]),
        generation=judge["lease_generation"], claim_hash=judge["claim_request_hash"])
    for path in (LIVE / "objects").rglob("*.json.gz"):
        objects.put(str(path.relative_to(LIVE / "objects")), path.read_bytes())
    if add_phase_states:
        # The injected live fixture did not run a full Arena agent. Add only
        # synthetic phase declarations; leave its real receipts and acks intact.
        snapshot = training_observations.read_snapshot(
            objects, run_id, "execute", execution["lease_generation"], execution["claim_request_hash"])
        payload = copy.deepcopy(snapshot["payload"])
        originals = copy.deepcopy((payload["response_receipts"], payload["agent_observations"]))
        for name in ("after_discovery", "after_verification", "after_admission", "final",
                     "training_capture_summary"):
            value = ({} if name != "training_capture_summary" else {
                "contract_version": 1, "scope": "agent_direct_egress_and_worker_socket",
                "sdk_available": True, "state_recorded": 4,
                "state_record_errors": 0, "state_local_errors": 0,
                "state_overflow_rows": 0, "state_overflow_manifests": 0})
            payload["agent_observations"].append({
                "id": len(payload["agent_observations"]), "kind": "state_snapshot",
                "observed_at": "2099-10-14T00:00:00Z", "after_action_sequence": None,
                "depends_on": [], "data": {"name": name, "value": value},
                "privacy": {"redacted": False, "redactions": []}})
        payload["agent_observation_count"] = len(payload["agent_observations"])
        assert payload["response_receipts"] == originals[0]
        assert payload["agent_observations"][:len(originals[1])] == originals[1]
        _, compressed = training_observations.build_snapshot(
            run_id, "execute", payload,
            lease_generation=execution["lease_generation"], claim_request_hash=execution["claim_request_hash"])
        objects.put(training_observations.object_ref(
            run_id, execution["lease_generation"], execution["claim_request_hash"]), compressed)
    values.update(run_id=run_id, ledger_rows=rows)
    return values


def test_real_provider_and_receipt_bytes_need_explicit_complete_context():
    original = assemble_example(**_context(add_phase_states=False))
    assert original["completeness"]["complete"] is False
    assert "agent_task_state_unverified" in original["completeness"]["incomplete_reasons"]
    complete_values = _context(add_phase_states=True)
    complete = assemble_example(**complete_values)
    assert complete["completeness"]["complete"] is True, complete["completeness"]["incomplete_reasons"]
    submission = complete_values["submission_row"]
    result = training_eligibility.assess_capture(
        submission, {complete["run_id"]: complete}, [complete["run_id"]], enabled=True)
    assert result["eligible"] is True, result
    assert result["counts"]["model_calls"] == 2
    assert result["counts"]["full_thinking_calls"] >= 1


def test_missing_saved_request_or_worker_receipt_fails_real_assembly():
    values = _context(add_phase_states=True)
    request_ref = values["ledger_rows"][0]["entry_doc"]["request_capture"]["ref"]
    del values["objects"].values[request_ref]
    example = assemble_example(**values)
    assert example["completeness"]["complete"] is False
    result = training_eligibility.assess_capture(
        values["submission_row"], {example["run_id"]: example}, [example["run_id"]], enabled=True)
    assert result["eligible"] is False


def test_removing_one_actual_agent_read_ack_fails_complete_export():
    values = _context(add_phase_states=True)
    execute = values["runs"][0]
    run_id = execute["run_id"]
    snapshot = training_observations.read_snapshot(
        values["objects"], run_id, "execute", execute["lease_generation"],
        execute["claim_request_hash"])
    payload = copy.deepcopy(snapshot["payload"])
    removed_identity = values["ledger_rows"][0]["call_identity"]
    payload["agent_observations"] = [row for row in payload["agent_observations"]
                                     if not (row["kind"] == "response_received"
                                             and row["data"]["call_identity"] == removed_identity)]
    for index, row in enumerate(payload["agent_observations"]):
        row["id"] = index
    payload["agent_observation_count"] = len(payload["agent_observations"])
    _, compressed = training_observations.build_snapshot(
        run_id, "execute", payload,
        lease_generation=execute["lease_generation"], claim_request_hash=execute["claim_request_hash"])
    values["objects"].put(training_observations.object_ref(
        run_id, execute["lease_generation"], execute["claim_request_hash"]), compressed)
    example = assemble_example(**values)
    assert example["completeness"]["complete"] is False
    assert "agent_response_receipts_unverified" in example["completeness"]["incomplete_reasons"]
    result = training_eligibility.assess_capture(
        values["submission_row"], {run_id: example}, [run_id], enabled=True)
    assert result["eligible"] is False
