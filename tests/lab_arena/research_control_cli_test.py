from __future__ import annotations

import json
from pathlib import Path

from lab_arena import research_control, source_bundle
from scripts import assess_arena_research_control as cli


def archive(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "source.tar.gz"
    source_bundle.write_source_archive(root / "tests/fixtures/research_control/fixed_extractor", output)
    return output


def response(source):
    prepared = research_control.prepare_request(source.read_bytes(), max_output_tokens=4096)
    document = {
        "schema_version": research_control.SCHEMA_VERSION,
        "policy_id": prepared.policy_id,
        "policy_hash": prepared.policy_hash,
        "source_archive_sha256": prepared.source_archive_sha256,
        "source_manifest_sha256": prepared.source_manifest_sha256,
        "classification": "workflow",
        "reviewed_files": list(prepared.reviewed_files),
        "evidence": [{"claim": "fixed_control", "file": "harness.py", "excerpt": "def run_icp(icp):",
                      "explanation": "Synthetic response for CLI validation, not a semantic judgment."}],
        "limitations": ["This mocked assessment does not prove classification accuracy."],
    }
    return {"model": prepared.parameters["model"], "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(document)}}]}


def test_prepare_has_private_exclusive_output_and_no_eligibility_claim(tmp_path, capsys):
    source, output = archive(tmp_path), tmp_path / "request.json"
    assert cli.main(["--source", str(source), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert output.stat().st_mode & 0o777 == 0o600
    assert value["production_eligibility"] == "not_established"
    assert value["runtime_behavior"] == "not_verified"
    assert value["parameters"]["provider"]["zdr"] is True
    before = output.read_bytes()
    assert cli.main(["--source", str(source), "--output", str(output)]) == 1
    assert output.read_bytes() == before
    assert "submission_files" not in capsys.readouterr().out


def test_parse_preserves_source_and_response(tmp_path):
    source, output, reply = archive(tmp_path), tmp_path / "assessment.json", tmp_path / "reply.json"
    reply.write_text(json.dumps(response(source)))
    before = (source.read_bytes(), reply.read_bytes())
    assert cli.main(["--source", str(source), "--response", str(reply), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["assessment"]["classification"] == "workflow"
    assert result["production_eligibility"] == "not_established"
    assert before == (source.read_bytes(), reply.read_bytes())


def test_duplicate_provider_json_and_truncated_responses_cannot_create_assessment(tmp_path):
    source, output, reply = archive(tmp_path), tmp_path / "assessment.json", tmp_path / "reply.json"
    reply.write_text('{"model":"one","model":"two"}')
    assert cli.main(["--source", str(source), "--response", str(reply), "--output", str(output)]) == 1
    assert not output.exists()
    value = response(source)
    value["choices"][0]["finish_reason"] = "length"
    reply.write_text(json.dumps(value))
    assert cli.main(["--source", str(source), "--response", str(reply), "--output", str(output)]) == 1
    assert not output.exists()


def test_symlink_output_does_not_overwrite_target(tmp_path):
    source, output, target = archive(tmp_path), tmp_path / "link.json", tmp_path / "keep.json"
    target.write_text("keep")
    output.symlink_to(target)
    assert cli.main(["--source", str(source), "--output", str(output)]) == 1
    assert target.read_text() == "keep"


def test_oversize_input_does_not_leak_or_create_output(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "source.tar.gz", tmp_path / "request.json"
    source.write_bytes(b"PRIVATE_SOURCE_PAYLOAD")
    monkeypatch.setattr(cli.source_bundle, "MAX_SOURCE_ARCHIVE_BYTES", 2)
    assert cli.main(["--source", str(source), "--output", str(output)]) == 1
    assert not output.exists()
    assert "PRIVATE_SOURCE" not in capsys.readouterr().err
