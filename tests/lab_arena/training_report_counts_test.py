"""Missing native report counters stay unknown in saved aggregates."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import aggregate_arena_capture_reports as aggregate


def _index(tmp_path: Path, captures: list[dict]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, capture in enumerate(captures):
        run_id = f"run-{position}"
        report_file = f"report-{position}.json"
        report = {
            "run_id": run_id,
            "capture_proof": capture,
            "full_text_characters": 0,
            "readable_path_proof": {"verified": True},
            "model_routes": {},
            "model_settings": {},
            "accounting": {},
        }
        (tmp_path / report_file).write_text(json.dumps(report))
        rows.append({
            "run_id": run_id,
            "status": "native_export_verified",
            "serving_sha": "source-sha",
            "selected_provider_calls": 1,
            "selected_model_calls": 1,
            "reasoning_tokens_reported": 0,
            "positive_tokens_without_full_text": 0,
            "readable_projection_truncated": False,
            "raw_example_hash": "raw-hash",
            "score_run_id": None,
            "judgment_present": False,
            "model_classes": {"complete": 1},
            "raw_incomplete_reasons": [],
            "report_file": report_file,
        })
    path = tmp_path / "index.json"
    path.write_text(json.dumps({
        "accepted_count": len(rows),
        "native_verified_count": len(rows),
        "missing_count": 0,
        "runs": rows,
    }))
    return path


def test_measured_zero_is_known_zero():
    measured = aggregate._capture_count(
        {"summary_model_calls": 0}, "summary_model_calls", "summary_calls"
    )
    assert measured == {
        "value": 0,
        "source_field": "summary_model_calls",
        "status": "measured",
    }


def test_all_missing_counts_have_null_totals_and_unknown_provenance(tmp_path):
    result = aggregate.aggregate_index(_index(tmp_path, [{}, {}]))
    summary = result["summary"]
    assert summary["summary_model_calls_known"] == 0
    assert summary["summary_model_calls_unknown_runs"] == 2
    assert summary["summary_model_calls_total"] is None
    assert summary["opaque_model_calls_known"] == 0
    assert summary["opaque_model_calls_unknown_runs"] == 2
    assert summary["opaque_model_calls_total"] is None
    assert result["capture_call_source_coverage"]["summary_model_calls"]["unknown_reasons"] == {"absent": 2}
    assert sum(
        item.get("field") == "summary_model_calls" and item.get("status") == "absent"
        for item in result["missing_fields"]
    ) == 2


def test_alias_and_measured_zero_preserve_partial_subtotal_but_total_is_unknown(tmp_path):
    captures = [
        {"summary_model_calls": 0, "opaque_model_calls": 0},
        {"summary_calls": 7, "opaque_calls": 2},
        {"summary_model_calls": None, "opaque_model_calls": "bad"},
        {"summary_model_calls": -1, "opaque_model_calls": True},
        {"summary_model_calls": 2, "summary_calls": 3, "opaque_model_calls": 4, "opaque_calls": 4},
    ]
    result = aggregate.aggregate_index(_index(tmp_path, captures))
    summary = result["summary"]
    assert summary["summary_model_calls_known"] == 7
    assert summary["summary_model_calls_unknown_runs"] == 3
    assert summary["summary_model_calls_total"] is None
    assert summary["opaque_model_calls_known"] == 6
    assert summary["opaque_model_calls_unknown_runs"] == 2
    assert summary["opaque_model_calls_total"] is None
    assert result["capture_call_source_coverage"]["summary_model_calls"]["source_fields"] == {
        "capture_proof.summary_calls": 1,
        "capture_proof.summary_model_calls": 1,
    }
    assert result["capture_call_source_coverage"]["summary_model_calls"]["unknown_reasons"] == {
        "conflicting_values": 1,
        "negative": 1,
        "null": 1,
    }


def test_cli_reads_explicit_index_and_writes_new_result_to_output_dir(tmp_path, capsys):
    index_path = _index(tmp_path / "input", [{"summary_model_calls": 0, "opaque_model_calls": 0}])
    output_dir = tmp_path / "results"
    assert aggregate.main(["--index", str(index_path), "--output-dir", str(output_dir)]) == 0
    response = json.loads(capsys.readouterr().out)
    output_path = Path(response["path"])
    written = json.loads(output_path.read_text())
    assert output_path.parent == output_dir
    assert written["summary"]["summary_model_calls_total"] == 0
    assert written["summary"]["opaque_model_calls_total"] == 0
    assert response["sha256"]


def test_output_does_not_overwrite_same_timestamp_result(tmp_path, monkeypatch):
    fixed_time = dt.datetime(2026, 10, 28, tzinfo=dt.timezone.utc)

    class FixedDateTime:
        @staticmethod
        def now(_timezone):
            return fixed_time

    monkeypatch.setattr(aggregate.datetime, "datetime", FixedDateTime)
    output_dir = tmp_path / "results"
    result = {"summary": {"summary_model_calls_total": None}}
    path = aggregate.write_aggregate(result, output_dir)
    original = path.read_text()
    with pytest.raises(FileExistsError):
        aggregate.write_aggregate({"different": True}, output_dir)
    assert path.read_text() == original


def test_mixed_report_containers_fallback_and_conflicts():
    read = lambda value: aggregate._report_capture_count(value, "summary_model_calls", "summary_calls")
    assert read({"capture_proof": {"unrelated": True}, "model_observations": {"summary_calls": 7}}) == {
        "value": 7, "source_field": "model_observations.summary_calls", "status": "measured"}
    assert read({"capture_proof": {"summary_calls": 7}, "model_observations": {"summary_calls": 8}})["status"] == "conflicting_values"
    assert read({"capture_proof": {"summary_calls": 7}, "model_observations": {"summary_calls": 7}})["value"] == 7
    assert read({"capture_proof": "invalid", "model_observations": {"summary_calls": 7}})["status"] == "bad_container"


def test_optimized_python_cannot_accept_incomplete_index(tmp_path):
    index = _index(tmp_path / "input", [{}])
    doc = json.loads(index.read_text())
    doc["missing_count"] = 1
    index.write_text(json.dumps(doc))
    result = subprocess.run([sys.executable, "-O", aggregate.__file__, "--index", str(index),
                             "--output-dir", str(tmp_path / "output")], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "output").exists()
