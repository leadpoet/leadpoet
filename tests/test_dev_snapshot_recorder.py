"""Failure-accounting tests for the development snapshot recorder."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from research_lab.eval.snapshot_store import (
    SNAPSHOT_MISS_SENTINEL,
    SNAPSHOT_RECORD_REUSE_EXISTING_ENV,
    dev_record_bootstrap,
)
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


def test_snapshot_runtime_context_finishes_before_host_timeout():
    context = recorder._snapshot_runtime_context(
        "dev_snapshot_recording",
        timeout_seconds=300,
    )

    assert context["dev_snapshot_recording"] is True
    options = context["runtime_options"]
    assert options["runtime_cap_seconds"] == 267.0
    assert options["finalization_reserve_seconds"] == pytest.approx(26.7)
    assert options["agent_timeout_seconds"] == 240
    assert options["runtime_cap_seconds"] <= 300 - 30


def test_named_docker_is_removed_when_host_run_is_interrupted(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, timeout=5)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(recorder.subprocess, "run", run)

    try:
        recorder._run_named_docker(
            ["docker", "run", "--name", "snapshot-test", "image"],
            container_name="snapshot-test",
            input_text="{}",
            timeout_seconds=5,
            environment={"PATH": "/usr/bin"},
        )
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("the host timeout must propagate")

    assert calls[1][0] == ["docker", "rm", "-f", "snapshot-test"]
    assert calls[1][1]["timeout"] == 30


@pytest.mark.parametrize("reuse_existing", [False, True])
def test_record_docker_only_enables_existing_snapshot_reuse_when_requested(
    monkeypatch,
    tmp_path,
    reuse_existing,
):
    captured = {}

    def run_named(command, **kwargs):
        captured["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="[]", stderr="receipt")

    monkeypatch.setattr(recorder, "_run_named_docker", run_named)
    monkeypatch.setattr(
        "research_lab.eval.private_runtime.validate_sourcing_runtime_receipt",
        lambda *args, **kwargs: {},
    )

    assert recorder._record_icp_with_docker(
        image_digest="example.invalid/model@sha256:" + "1" * 64,
        module_name="research_lab_adapter",
        callable_name="run_icp",
        icp={
            "industry": "Software Development",
            "intent_signal": "hiring sales leadership",
        },
        snapshot_dir=str(tmp_path),
        timeout_seconds=300,
        reuse_existing=reuse_existing,
    ) == []

    docker_environment = [
        captured["command"][index + 1]
        for index, argument in enumerate(captured["command"][:-1])
        if argument == "-e"
    ]
    expected = f"{SNAPSHOT_RECORD_REUSE_EXISTING_ENV}=true"
    assert (expected in docker_environment) is reuse_existing


def test_snapshot_closure_replays_only_icps_that_expose_new_requests(monkeypatch):
    class Store:
        count = 10

        def snapshot_count(self):
            return self.count

    store = Store()
    calls = []

    def record(**kwargs):
        calls.append(kwargs["icp_ref"])
        if kwargs["icp_ref"] == "icp-a" and calls.count("icp-a") == 1:
            store.count += 1
        assert kwargs["reuse_existing"] is True
        return []

    monkeypatch.setattr(recorder, "_record_icp_with_docker", record)
    result = recorder._close_snapshot_request_set(
        items=[
            {"icp_ref": "icp-a", "icp": {"industry": "Software"}},
            {"icp_ref": "icp-b", "icp": {"industry": "Healthcare"}},
        ],
        store=store,
        image_digest="example.invalid/model@sha256:" + "1" * 64,
        module_name="research_lab_adapter",
        callable_name="run_icp",
        snapshot_dir="/tmp/snapshot-test",
        timeout_seconds=300,
    )

    assert result == {
        "stable": True,
        "rounds": 2,
        "pending_icp_count": 0,
        "runner_failure_refs": [],
    }
    assert calls == ["icp-a", "icp-b", "icp-a"]


def test_snapshot_closure_fails_closed_when_request_set_never_stabilizes(
    monkeypatch,
):
    class Store:
        count = 0

        def snapshot_count(self):
            return self.count

    store = Store()

    def record(**kwargs):
        assert kwargs["reuse_existing"] is True
        store.count += 1
        return []

    monkeypatch.setattr(recorder, "_record_icp_with_docker", record)
    result = recorder._close_snapshot_request_set(
        items=[{"icp_ref": "icp-a", "icp": {"industry": "Software"}}],
        store=store,
        image_digest="example.invalid/model@sha256:" + "1" * 64,
        module_name="research_lab_adapter",
        callable_name="run_icp",
        snapshot_dir="/tmp/snapshot-test",
        timeout_seconds=300,
        max_rounds=2,
    )

    assert result == {
        "stable": False,
        "rounds": 2,
        "pending_icp_count": 1,
        "runner_failure_refs": [],
    }
    assert recorder._recording_is_complete(
        closure_result=result,
        failure_summary={"has_failures": False},
    ) is False


def test_snapshot_recording_is_complete_only_after_closure_without_failures():
    stable = {"stable": True}

    assert recorder._recording_is_complete(
        closure_result=stable,
        failure_summary={"has_failures": False},
    ) is True
    assert recorder._recording_is_complete(
        closure_result=stable,
        failure_summary={"has_failures": True},
    ) is False


def test_snapshot_closure_reports_runner_failure(monkeypatch):
    class Store:
        def snapshot_count(self):
            return 0

    def fail(**_kwargs):
        raise RuntimeError("bounded recorder failure")

    monkeypatch.setattr(recorder, "_record_icp_with_docker", fail)
    result = recorder._close_snapshot_request_set(
        items=[{"icp_ref": "icp-a", "icp": {"industry": "Software"}}],
        store=Store(),
        image_digest="example.invalid/model@sha256:" + "1" * 64,
        module_name="research_lab_adapter",
        callable_name="run_icp",
        snapshot_dir="/tmp/snapshot-test",
        timeout_seconds=300,
    )

    assert result["stable"] is False
    assert result["runner_failure_refs"] == ["icp-a"]


def test_offline_replay_uses_only_nonsecret_key_sentinels(monkeypatch, tmp_path):
    captured = {}

    def run_named(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(returncode=0, stdout="[]", stderr="receipt")

    monkeypatch.setattr(recorder, "_run_named_docker", run_named)
    monkeypatch.setattr(
        "research_lab.eval.private_runtime.validate_sourcing_runtime_receipt",
        lambda *args, **kwargs: {},
    )

    assert recorder._replay_icp_with_docker(
        image_digest="example.invalid/model@sha256:" + "1" * 64,
        module_name="research_lab_adapter",
        callable_name="run_icp",
        icp={
            "industry": "Software Development",
            "intent_signal": "hiring sales leadership",
        },
        snapshot_dir=str(tmp_path),
        timeout_seconds=300,
    ) == []

    command = captured["command"]
    assert command[command.index("--network") + 1] == "none"
    replay_environment = [
        command[index + 1]
        for index, argument in enumerate(command[:-1])
        if argument == "-e"
    ]
    for group in recorder.PROVIDER_KEY_GROUPS:
        for name in group:
            assert f"{name}=research-lab-offline-replay" in replay_environment
    assert all("sk-or-" not in value.lower() for value in replay_environment)


def test_offline_replay_rejects_caught_strict_snapshot_miss(monkeypatch, tmp_path):
    def run_named(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="[]",
            stderr=(
                "receipt\n"
                + SNAPSHOT_MISS_SENTINEL
                + "exa|GET|api.exa.ai/search|sha256:redacted\n"
            ),
        )

    monkeypatch.setattr(recorder, "_run_named_docker", run_named)

    with pytest.raises(
        RuntimeError,
        match="offline replay observed a strict snapshot miss",
    ):
        recorder._replay_icp_with_docker(
            image_digest="example.invalid/model@sha256:" + "1" * 64,
            module_name="research_lab_adapter",
            callable_name="run_icp",
            icp={
                "industry": "Software Development",
                "intent_signal": "hiring sales leadership",
            },
            snapshot_dir=str(tmp_path),
            timeout_seconds=300,
        )


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
