"""Failure-accounting tests for the development snapshot recorder."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from research_lab.eval.snapshot_store import dev_record_bootstrap
from scripts import record_research_lab_dev_snapshots as recorder


def test_recording_failure_summary_deduplicates_events_and_icps(tmp_path):
    failure_file = tmp_path / "record_failures.jsonl"
    rows = [
        {"icp_ref": "icp-a", "request_key": "key-a", "reason": "write_error"},
        {"icp_ref": "icp-a", "request_key": "key-a", "reason": "write_error"},
        {"icp_ref": "icp-b", "request_key": "key-b", "reason": "secret_rejected"},
        {"icp_ref": "", "request_key": "key-c", "reason": "install_error"},
    ]
    failure_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\nnot-json\n",
        encoding="utf-8",
    )

    summary = recorder._recording_failure_summary(
        runner_failure_refs=["icp-a", "icp-a", "icp-c"],
        failure_file=failure_file,
    )

    assert summary == {
        "runner_failure_count": 2,
        "provider_failure_event_count": 4,
        "failed_icp_count": 3,
        "unattributed_provider_failure_count": 2,
        "has_failures": True,
    }


def test_recording_failure_summary_is_empty_when_no_failures(tmp_path):
    summary = recorder._recording_failure_summary(
        runner_failure_refs=[],
        failure_file=tmp_path / "missing.jsonl",
    )

    assert summary["has_failures"] is False
    assert summary["failed_icp_count"] == 0
    assert summary["provider_failure_event_count"] == 0


def test_recording_bootstrap_attributes_failure_to_current_icp(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/usr/bin")

    env = recorder._subprocess_env(str(tmp_path), icp_ref="icp-8")
    bootstrap = dev_record_bootstrap()

    assert env["RESEARCH_LAB_DEV_RECORD_ICP_REF"] == "icp-8"
    assert '"icp_ref": _RL_DEV_RECORD_ICP_REF[:500]' in bootstrap


def test_recording_bootstrap_captures_actual_openrouter_model(tmp_path):
    script = dev_record_bootstrap() + """
assert _rl_dev_record(
    "POST",
    "https://openrouter.ai/api/v1/chat/completions",
    {"model": "openai/gpt-production", "messages": [{"role": "user", "content": "x"}]},
    200,
    {"content-type": "application/json"},
    '{"choices": []}',
)
"""
    env = {
        **os.environ,
        "RESEARCH_LAB_DEV_SNAPSHOT_DIR": str(tmp_path),
        "RESEARCH_LAB_DEV_RECORD_ICP_REF": "icp-1",
    }
    subprocess.run([sys.executable, "-c", script], env=env, check=True)

    assert recorder._recorded_provider_model_ids(Path(tmp_path)) == [
        "openai/gpt-production"
    ]
    assert not (tmp_path / "provider_models.jsonl").exists()


def test_observed_provider_models_are_authoritative():
    assert recorder._resolve_provider_model_ids(
        ["openai/model-b", "openai/model-a", "openai/model-a"], []
    ) == ["openai/model-a", "openai/model-b"]

    try:
        recorder._resolve_provider_model_ids(
            ["openai/unexpected"], ["openai/expected"]
        )
    except ValueError as exc:
        assert "outside the declared allowlist" in str(exc)
    else:
        raise AssertionError("an unexpected observed model must fail closed")
