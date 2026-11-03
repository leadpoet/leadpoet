"""Independent checks of private readiness inventory and report publication."""
import copy
import json
import stat

import pytest

from lab_arena import training_readiness as readiness
from scripts import report_arena_training as cli
from tests.lab_arena.training_dataset_test import example, claim, row, seal
from tests.lab_arena.training_eligibility_test import ten_selected_examples


def document(run="private-run"):
    return example(run, [row(0, claim(), "qualified")])


def test_real_build_reports_company_candidate_without_claiming_complete_execution():
    report = readiness.build([document()])
    assert report["qualification"]["candidate_records"] == 1
    assert report["capture"]["executions_passing_all_export_dimensions"] == 0
    assert report["capture"]["passing_dimensions"]["input_output_label"] == 1
    assert report["readiness"]["training_ready"] is False
    assert report["readiness"]["human_label_review"] == "not_proved_by_storage"
    assert report["agent_steps"]["candidate_records"] == 0


def test_full_thinking_inventory_reports_frozen_coverage_without_champion_claim():
    exports = list(ten_selected_examples().values())
    for position, item in enumerate(exports):
        item["configuration"]["training_minimum_icp_coverage_percent"] = 80
        item["configuration"].update(stage_1_icp_count=5, stage_2_icp_count=5)
        item["icp_position"] = position
        seal(item)
    report = readiness.build(exports, production_rounds_only=False)
    coverage = report["full_thinking"]["coverage"]
    assert len(coverage) == 1
    assert coverage[0]["provided_icps"] == 10
    assert coverage[0]["required_icps"] == 8
    assert coverage[0]["strict_icps"] == 10
    assert coverage[0]["supplied_inventory_complete"] is True
    assert coverage[0]["strict_coverage_met_in_supplied_inventory"] is True
    assert report["readiness"]["training_ready"] is False
    exports[-1]["completeness"].update(complete=False, incomplete_reasons=["response_privacy_nonexact"])
    report = readiness.build([*exports[:-1], seal(exports[-1])], production_rounds_only=False)
    assert report["full_thinking"]["coverage"][0]["strict_icps"] == 10
    # Other exact pairs remain in the export inventory. The eligibility gate
    # separately rejects an ICP that has a real selected-result redaction.
    call = exports[-1]["provider_calls"][1]
    call["response_privacy_redactions"] = ["credential"]
    call["response_receipts"][0]["privacy_redactions"] = ["credential"]
    report = readiness.build([*exports[:-1], seal(exports[-1])], production_rounds_only=False)
    assert report["full_thinking"]["coverage"][0]["strict_icps"] == 10
    assert "not_complete_icps" in report["full_thinking"]["coverage_scope"]
    from lab_arena import training_eligibility, training_steps, training_thinking
    from tests.lab_arena.training_eligibility_test import submission
    example = exports[-1]
    gate = training_eligibility.assess_capture(
        submission(), {example["run_id"]: example}, [example["run_id"]], enabled=True)
    assert gate["eligible"] is False
    splits, _ = training_thinking.build([example], training_steps.Rules(production_rounds_only=False))
    records = splits["train"] + splits["eval"]
    assert len(records) == 1
    assert all(ref["call_identity"] != call["call_identity"] for row in records for ref in row["source_refs"])


def test_duplicate_exports_conflicting_versions_and_bad_hash_are_not_counted_twice():
    first = document()
    version = copy.deepcopy(first)
    version["privacy"]["context_redactions"] = ["changed metadata"]
    seal(version)
    invalid = document("bad")
    invalid["evaluation_date"] = "changed without seal"
    report = readiness.build([first, first, version, invalid])
    assert report["distinct_verified_executions"] == 0
    assert report["excluded_exports"] == {"duplicate_export": 1, "conflicting_export_versions": 2,
                                           "example_hash_mismatch": 1}
    assert report["qualification"]["candidate_records"] == 0
    assert report == readiness.build([invalid, version, first, first])


def test_mixed_scorer_cohorts_are_separate_instead_of_combined():
    first, second = document("one"), document("two")
    second["configuration"]["scorer_image_digest"] = "sha256:" + "c" * 64
    report = readiness.build([first, seal(second)])
    assert len(report["qualification"]["cohorts"]) == 2
    assert report["qualification"]["candidate_records"] == 2
    assert report["qualification"]["cohorts_must_not_be_combined_for_training"]


def test_missing_image_cohort_stays_exact_while_using_full_source_split_graph():
    first = document("image-missing")
    first["configuration"]["scorer_image_digest"] = None
    first["submission_id"] = "missing-image-source"
    first["companies"][0]["company"] = claim("Missing Image", "missing-image.example", "missing-image")
    first["output"]["companies"][0] = first["companies"][0]["company"]
    second = document("image-present")
    second["submission_id"] = "present-image-source"
    second["companies"][0]["company"] = claim("Present Image", "present-image.example", "present-image")
    second["output"]["companies"][0] = second["companies"][0]["company"]
    report = readiness.build([seal(first), seal(second)])
    cohorts = report["qualification"]["cohorts"]
    assert len(cohorts) == 2
    assert [sum(cohort["records"][name] for name in ("train", "eval"))
            for cohort in cohorts] == [1, 1]
    assert report["qualification"]["candidate_records"] == 2


def test_truthy_completeness_metadata_cannot_claim_complete():
    value = document()
    value["completeness"] = {"complete": True, "dimensions": {
        key: "true" for key in readiness._DIMENSIONS}, "incomplete_reasons": []}
    report = readiness.build([seal(value)])
    assert report["capture"]["executions_passing_all_export_dimensions"] == 0
    assert not any(report["capture"]["passing_dimensions"].values())
    assert report["qualification"]["candidate_records"] == 0


def test_per_run_status_uses_joined_proof_not_successful_recorder_buffer():
    value = document()
    value["worker_observation_artifacts"] = {value["run_id"]: {"payload": {
        "complete": True, "agent_observations": [{"kind": "state_snapshot", "data": {
            "name": "automatic_capture_summary", "value": {"counts": {}, "complete": True}}}]}}}
    value["provider_calls"] = [
        {"run_id": value["run_id"], "response_provenance": {
            "delivery_to_worker": "socket_written", "agent_receipt": "not_reported"}},
        {"run_id": value["run_id"], "response_provenance": {
            "delivery_to_worker": "socket_written", "agent_receipt": "reported_exact_body_received"}},
        {"run_id": "prior-attempt", "response_provenance": {"delivery_to_worker": "socket_written"}},
    ]
    report = readiness.build([seal(value)])
    status = report["sources"][0]["capture"]
    assert status["status"] == "incomplete"
    assert status["authority"] == "joined_export_checks"
    assert status["recorder_buffer_complete"] is True
    assert status["response_receipts"] == {
        "socket_written": 2, "read_reported": 1, "written_without_read_report": 1}
    assert "direct_web_content_not_captured" in status["incomplete_reasons"]
    assert report["qualification"]["candidate_records"] == 1


def test_per_run_status_keeps_old_missing_proof_unverified():
    value = document()
    value.pop("completeness")
    status = readiness.build([seal(value)])["sources"][0]["capture"]
    assert status["status"] == "unverified"
    assert status["recorder_buffer_complete"] is None


def test_per_run_complete_requires_every_export_dimension():
    value = document()
    value["completeness"] = {"complete": True, "dimensions": {
        key: True for key in readiness._DIMENSIONS}, "incomplete_reasons": []}
    report = readiness.build([seal(value)])
    assert report["sources"][0]["capture"]["status"] == "complete"
    value["completeness"]["dimensions"]["research_process"] = False
    assert readiness.build([seal(value)])["sources"][0]["capture"]["status"] == "incomplete"


def test_cli_private_report_and_no_private_payload_in_stdout(tmp_path, capsys):
    source, output = tmp_path / "export.json", tmp_path / "report.json"
    value = document("private-execution-identity")
    source.write_text(json.dumps(value))
    assert cli.main(["--examples", str(source), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    for private in ("private-execution-identity", "Acme", "company_qualified", "verifier_gate_receipts"):
        assert private not in captured.out
    summary = json.loads(captured.out)
    report = json.loads(output.read_text())
    assert summary["report_hash"] == report["report_hash"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.parametrize("symlink", [False, True])
def test_cli_never_overwrites_existing_file_or_symlink(tmp_path, capsys, symlink):
    source, output = tmp_path / "export.json", tmp_path / "report.json"
    source.write_text(json.dumps(document()))
    target = tmp_path / "existing.json"
    target.write_text("existing private data")
    if symlink:
        output.symlink_to(target)
    else:
        output.write_text("existing private data")
    assert cli.main(["--examples", str(source), "--output", str(output)]) == 1
    assert output.read_text() == target.read_text() == "existing private data"
    assert capsys.readouterr().err == "Arena training report failed: FileExistsError\n"


def test_cli_malformed_metadata_fails_without_public_payload(tmp_path, capsys):
    source, output = tmp_path / "export.json", tmp_path / "report.json"
    value = document()
    value["completeness"] = {"complete": True, "dimensions": "private malformed metadata"}
    source.write_text(json.dumps(seal(value)))
    assert cli.main(["--examples", str(source), "--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert "private malformed" not in captured.err
    assert not output.exists()


def test_cli_removes_only_its_new_partial_report_after_write_failure(tmp_path, monkeypatch, capsys):
    from scripts import prepare_arena_training_dataset as writer
    source, output = tmp_path / "export.json", tmp_path / "report.json"
    source.write_text(json.dumps(document()))
    fdopen = writer.os.fdopen

    class FailedStream:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def write(self, text):
            self.stream.write(text[:12])
            self.stream.flush()
            raise OSError("private source payload")
        def __exit__(self, *args):
            self.stream.close()

    monkeypatch.setattr(writer.os, "fdopen", lambda *args, **kwargs: FailedStream(fdopen(*args, **kwargs)))
    assert cli.main(["--examples", str(source), "--output", str(output)]) == 1
    assert not output.exists()
    assert capsys.readouterr().err == "Arena training report failed: OSError\n"
