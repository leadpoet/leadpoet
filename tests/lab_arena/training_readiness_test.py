"""Independent checks of private readiness inventory and report publication."""
import copy
import json
import stat

import pytest

from lab_arena import training_readiness as readiness
from scripts import report_arena_training as cli
from tests.lab_arena.training_dataset_test import example, claim, row, seal


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
