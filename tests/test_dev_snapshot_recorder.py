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
    DevSnapshotStoreError,
    MODE_RECORD,
    SNAPSHOT_MISS_SENTINEL,
    SNAPSHOT_RECORD_RETRY_TRANSIENT_ENV,
    SNAPSHOT_RECORD_REUSE_EXISTING_ENV,
    ProviderSnapshotStore,
    build_snapshot_request,
    dev_record_bootstrap,
)
from scripts import record_research_lab_dev_snapshots as recorder


def _artifact_document() -> dict[str, object]:
    return {
        "model_artifact_hash": "sha256:" + "1" * 64,
        "git_commit_sha": "2" * 40,
        "image_digest": "example.invalid/model@sha256:" + "3" * 64,
        "config_hash": "sha256:" + "4" * 64,
        "component_registry_version": "registry-v1",
        "scoring_adapter_version": "adapter-v1",
        "manifest_uri": "s3://private/model.json",
        "manifest_hash": "sha256:" + "5" * 64,
        "signature_ref": "kms:test",
    }


def test_snapshot_adapter_authority_selects_artifact_bound_bootstrap(
    monkeypatch, tmp_path
):
    artifact = _artifact_document()
    receipt = {"admission_mode": "qualification_protocol_v2"}
    artifact_path = tmp_path / "artifact.json"
    receipt_path = tmp_path / "receipt.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    observed = {}

    def select(value, *, artifact):
        observed["receipt"] = dict(value)
        observed["artifact"] = artifact.to_dict()
        return "selected-v2-bootstrap"

    monkeypatch.setattr(
        "gateway.tee.model_sandbox_v2."
        "_model_adapter_bootstrap_for_compatibility_receipt_v1",
        select,
    )

    assert recorder._load_snapshot_adapter_authority(
        artifact_path=str(artifact_path),
        compatibility_receipt_path=str(receipt_path),
        image_digest=str(artifact["image_digest"]),
        source_commit=str(artifact["git_commit_sha"]),
        model_config_hash=str(artifact["config_hash"]),
        manifest_hash=str(artifact["manifest_hash"]),
    ) == ("selected-v2-bootstrap", "qualification_protocol_v2")
    assert observed["receipt"] == receipt
    assert {
        key: observed["artifact"][key] for key in artifact
    } == artifact


def test_snapshot_adapter_authority_rejects_identity_drift(tmp_path):
    artifact = _artifact_document()
    artifact_path = tmp_path / "artifact.json"
    receipt_path = tmp_path / "receipt.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    receipt_path.write_text(
        json.dumps({"admission_mode": "qualification_protocol_v2"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from recorder identity"):
        recorder._load_snapshot_adapter_authority(
            artifact_path=str(artifact_path),
            compatibility_receipt_path=str(receipt_path),
            image_digest=str(artifact["image_digest"]),
            source_commit="6" * 40,
            model_config_hash=str(artifact["config_hash"]),
            manifest_hash=str(artifact["manifest_hash"]),
        )


def test_v2_snapshot_output_requires_complete_typed_outcome(monkeypatch):
    monkeypatch.setattr(
        "research_lab.eval.private_runtime.validate_qualification_outcome_envelope_v2",
        lambda value: dict(value),
    )
    complete = {
        "completion_state": "complete",
        "companies": [{"company_name": "Example"}],
    }
    assert recorder._decode_adapter_companies(
        complete,
        admission_mode="qualification_protocol_v2",
    ) == complete["companies"]

    with pytest.raises(RuntimeError, match="qualification outcome is incomplete"):
        recorder._decode_adapter_companies(
            {"completion_state": "incomplete", "companies": []},
            admission_mode="qualification_protocol_v2",
        )


def test_legacy_snapshot_output_remains_a_company_array():
    companies = [{"company_name": "Example"}]
    assert recorder._decode_adapter_companies(companies, admission_mode="legacy") == companies


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


def test_recording_bootstrap_allows_openrouter_control_without_model(tmp_path):
    script = dev_record_bootstrap() + """
assert _rl_dev_record(
    "GET",
    "https://openrouter.ai/api/v1/key",
    None,
    200,
    {"content-type": "application/json"},
    '{}',
)
"""
    env = {
        **os.environ,
        "RESEARCH_LAB_DEV_SNAPSHOT_DIR": str(tmp_path),
        "RESEARCH_LAB_DEV_RECORD_ICP_REF": "icp-1",
    }
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
    store = ProviderSnapshotStore(str(tmp_path), mode=MODE_RECORD)

    assert store.provider_request_counts() == {"openrouter": 1}
    assert store.provider_model_request_counts() == {}
    assert recorder._recorded_provider_model_ids(Path(tmp_path)) == []
    assert recorder._resolve_snapshot_provider_model_ids(
        store=store,
        observed=[],
        declared=[],
    ) == []


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


def test_full_bank_recording_timeout_covers_every_bounded_sequential_stage():
    assert recorder.snapshot_record_workflow_timeout_seconds(
        item_count=40,
        item_timeout_seconds=recorder.DEFAULT_SNAPSHOT_ICP_TIMEOUT_SECONDS,
    ) == 1_049_100

    for invalid in (0, -1, True):
        with pytest.raises(ValueError):
            recorder.snapshot_record_workflow_timeout_seconds(
                item_count=invalid,
                item_timeout_seconds=recorder.DEFAULT_SNAPSHOT_ICP_TIMEOUT_SECONDS,
            )
    with pytest.raises(ValueError):
        recorder.snapshot_record_workflow_timeout_seconds(
            item_count=40,
            item_timeout_seconds=0,
        )


def test_snapshot_export_bank_size_matches_full_exported_items(tmp_path):
    source = tmp_path / "source_icps.json"
    document = {
        "schema_version": "research_lab.dev_icp_export.v2",
        "items": [{"icp_ref": f"icp-{index}"} for index in range(40)],
        "daily_bank_manifest": {"bank_size": 40},
    }
    source.write_text(json.dumps(document), encoding="utf-8")
    assert recorder.snapshot_export_bank_size(str(source)) == 40

    document["daily_bank_manifest"]["bank_size"] = 5
    source.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="bank size differs"):
        recorder.snapshot_export_bank_size(str(source))


def test_named_docker_is_removed_when_host_run_is_interrupted(
    monkeypatch,
    tmp_path,
):
    calls = []

    def run(command, **kwargs):
        if list(command)[-1:] == ["info"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        calls.append((list(command), dict(kwargs)))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, timeout=5)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv(
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        str(tmp_path / "docker-operation.lock"),
    )
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
    assert calls[1][1]["timeout"] == (
        recorder.SNAPSHOT_DOCKER_CLEANUP_TIMEOUT_SECONDS
    )
def test_recorder_sigterm_unwinds_through_named_docker_cleanup(
    monkeypatch,
    tmp_path,
):
    calls = []

    def run(command, **kwargs):
        if list(command)[-1:] == ["info"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        calls.append((list(command), dict(kwargs)))
        if len(calls) == 1:
            recorder._terminate_snapshot_recorder_on_signal(
                recorder.signal.SIGTERM,
                None,
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv(
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        str(tmp_path / "docker-operation.lock"),
    )
    monkeypatch.setattr(recorder.subprocess, "run", run)

    with pytest.raises(SystemExit) as exc_info:
        recorder._run_named_docker(
            ["docker", "run", "--name", "snapshot-test", "image"],
            container_name="snapshot-test",
            input_text="{}",
            timeout_seconds=5,
            environment={"PATH": "/usr/bin"},
        )

    assert exc_info.value.code == 128 + int(recorder.signal.SIGTERM)
    assert calls[1][0] == ["docker", "rm", "-f", "snapshot-test"]


@pytest.mark.parametrize(
    ("reuse_existing", "retry_transient"),
    [(False, False), (True, False), (True, True)],
)
def test_record_docker_only_enables_existing_snapshot_reuse_when_requested(
    monkeypatch,
    tmp_path,
    reuse_existing,
    retry_transient,
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
    monkeypatch.setattr(
        recorder,
        "_decode_adapter_companies",
        lambda *args, **kwargs: [],
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
        retry_transient=retry_transient,
        adapter_bootstrap="selected-v2-bootstrap",
        admission_mode="qualification_protocol_v2",
    ) == []

    docker_environment = [
        captured["command"][index + 1]
        for index, argument in enumerate(captured["command"][:-1])
        if argument == "-e"
    ]
    expected = f"{SNAPSHOT_RECORD_REUSE_EXISTING_ENV}=true"
    assert (expected in docker_environment) is reuse_existing
    retry_expected = f"{SNAPSHOT_RECORD_RETRY_TRANSIENT_ENV}=true"
    assert (retry_expected in docker_environment) is retry_transient
    assert "LEADPOET_QUALIFICATION_PROTOCOL_V2=1" in docker_environment
    assert any("selected-v2-bootstrap" in argument for argument in captured["command"])


def test_snapshot_record_retry_reuses_partial_immutable_capture(
    monkeypatch,
) -> None:
    calls = []
    delays = []

    def record(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("transient container failure")
        return [{"company": "accepted"}]

    monkeypatch.setattr(recorder, "_record_icp_with_docker", record)
    monkeypatch.setattr(recorder.time, "sleep", delays.append)

    result = recorder._record_icp_with_retries(
        image_digest="example.invalid/model@sha256:" + "1" * 64,
        module_name="research_lab_adapter",
        callable_name="run_icp",
        icp={"industry": "Software"},
        icp_ref="icp-a",
        snapshot_dir="/tmp/snapshot-test",
        timeout_seconds=300,
        reuse_existing=False,
        item_index=1,
        item_count=5,
    )

    assert result == [{"company": "accepted"}]
    assert [call["reuse_existing"] for call in calls] == [False, True]
    assert [call["retry_transient"] for call in calls] == [False, True]
    assert delays == [5.0]


def test_snapshot_record_retry_remains_bounded_and_fail_closed(
    monkeypatch,
) -> None:
    calls = []
    delays = []

    def fail(**kwargs):
        calls.append(dict(kwargs))
        raise RuntimeError("persistent container failure")

    monkeypatch.setattr(recorder, "_record_icp_with_docker", fail)
    monkeypatch.setattr(recorder.time, "sleep", delays.append)

    with pytest.raises(RuntimeError, match="persistent container failure"):
        recorder._record_icp_with_retries(
            image_digest="example.invalid/model@sha256:" + "1" * 64,
            module_name="research_lab_adapter",
            callable_name="run_icp",
            icp={"industry": "Software"},
            icp_ref="icp-a",
            snapshot_dir="/tmp/snapshot-test",
            timeout_seconds=300,
            reuse_existing=False,
            item_index=1,
            item_count=5,
        )

    assert [call["reuse_existing"] for call in calls] == [False, True, True]
    assert [call["retry_transient"] for call in calls] == [False, True, True]
    assert delays == [5.0, 15.0]


def test_snapshot_record_retry_stops_at_boundary_when_superseded(
    monkeypatch,
    tmp_path,
) -> None:
    cancel_file = tmp_path / "cancel-recording"
    calls = []
    delays = []

    def fail_after_model_change(**kwargs):
        calls.append(dict(kwargs))
        cancel_file.write_text("active_private_model_changed\n", encoding="utf-8")
        raise RuntimeError("transient provider failure")

    monkeypatch.setattr(
        recorder,
        "_record_icp_with_docker",
        fail_after_model_change,
    )
    monkeypatch.setattr(recorder.time, "sleep", delays.append)

    with pytest.raises(
        recorder.SnapshotRecordingCancelled,
        match="active_private_model_changed",
    ):
        recorder._record_icp_with_retries(
            image_digest="example.invalid/model@sha256:" + "1" * 64,
            module_name="research_lab_adapter",
            callable_name="run_icp",
            icp={"industry": "Software"},
            icp_ref="icp-a",
            snapshot_dir="/tmp/snapshot-test",
            timeout_seconds=300,
            reuse_existing=False,
            item_index=1,
            item_count=5,
            cancel_file=cancel_file,
        )

    assert len(calls) == 1
    assert delays == []


def test_snapshot_closure_stops_before_next_icp_when_superseded(
    monkeypatch,
    tmp_path,
) -> None:
    cancel_file = tmp_path / "cancel-recording"
    calls = []

    class Store:
        def snapshot_count(self):
            return 10

    def record(**kwargs):
        calls.append(kwargs["icp_ref"])
        cancel_file.write_text("active_private_model_changed\n", encoding="utf-8")
        return []

    monkeypatch.setattr(recorder, "_record_icp_with_docker", record)

    with pytest.raises(recorder.SnapshotRecordingCancelled):
        recorder._close_snapshot_request_set(
            items=[
                {"icp_ref": "icp-a", "icp": {"industry": "Software"}},
                {"icp_ref": "icp-b", "icp": {"industry": "Healthcare"}},
            ],
            store=Store(),
            image_digest="example.invalid/model@sha256:" + "1" * 64,
            module_name="research_lab_adapter",
            callable_name="run_icp",
            snapshot_dir="/tmp/snapshot-test",
            timeout_seconds=300,
            cancel_file=cancel_file,
        )

    assert calls == ["icp-a"]


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
    monkeypatch.setattr(recorder.time, "sleep", lambda _seconds: None)
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
    monkeypatch.setattr(
        recorder,
        "_decode_adapter_companies",
        lambda *args, **kwargs: [],
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
        admission_mode="qualification_protocol_v2",
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
    assert "LEADPOET_QUALIFICATION_PROTOCOL_V2=1" in replay_environment
    assert all("sk-or-" not in value.lower() for value in replay_environment)


def test_record_and_offline_replay_use_identical_model_context(monkeypatch, tmp_path):
    payloads = []

    def run_named(_command, **kwargs):
        payloads.append(json.loads(kwargs["input_text"]))
        return SimpleNamespace(returncode=0, stdout="[]", stderr="receipt")

    monkeypatch.setattr(recorder, "_run_named_docker", run_named)
    monkeypatch.setattr(
        "research_lab.eval.private_runtime.validate_sourcing_runtime_receipt",
        lambda *args, **kwargs: {},
    )
    common = {
        "image_digest": "example.invalid/model@sha256:" + "1" * 64,
        "module_name": "research_lab_adapter",
        "callable_name": "run_icp",
        "icp": {
            "industry": "Software Development",
            "intent_signal": "hiring sales leadership",
        },
        "snapshot_dir": str(tmp_path),
        "timeout_seconds": 300,
    }

    recorder._record_icp_with_docker(icp_ref="icp-1", **common)
    recorder._replay_icp_with_docker(**common)

    assert len(payloads) == 2
    assert payloads[0]["context"] == payloads[1]["context"]
    assert payloads[0]["context"][recorder.SNAPSHOT_EXECUTION_CONTEXT_MARKER] is True
    assert "dev_snapshot_replay_validation" not in payloads[1]["context"]


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


def _store_with_provider_request(tmp_path, provider: str) -> ProviderSnapshotStore:
    store = ProviderSnapshotStore(str(tmp_path), mode=MODE_RECORD)
    if provider == "openrouter":
        request = build_snapshot_request(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            body={"model": "openai/gpt-production", "messages": []},
        )
    else:
        request = build_snapshot_request(
            "POST",
            "https://api.exa.ai/search",
            body={"query": "production-shaped query"},
        )
    store.record_response(request, status=200, body_text='{"results":[]}')
    return store


def _store_with_openrouter_control_request(
    tmp_path,
    *,
    method="GET",
    path="/api/v1/key",
) -> ProviderSnapshotStore:
    store = ProviderSnapshotStore(str(tmp_path), mode=MODE_RECORD)
    request = build_snapshot_request(
        method,
        "https://openrouter.ai" + path,
        body=None,
    )
    store.record_response(request, status=200, body_text="{}")
    return store


def test_snapshot_provider_models_are_optional_without_openrouter_requests(tmp_path):
    store = _store_with_provider_request(tmp_path, "exa")

    assert recorder._resolve_snapshot_provider_model_ids(
        store=store,
        observed=[],
        declared=[],
    ) == []


def test_snapshot_provider_models_remain_required_for_openrouter_requests(tmp_path):
    store = _store_with_provider_request(tmp_path, "openrouter")

    with pytest.raises(
        ValueError,
        match="champion emitted no attributable OpenRouter model request",
    ):
        recorder._resolve_snapshot_provider_model_ids(
            store=store,
            observed=[],
            declared=[],
        )

    assert recorder._resolve_snapshot_provider_model_ids(
        store=store,
        observed=["openai/gpt-production"],
        declared=["openai/gpt-production"],
    ) == ["openai/gpt-production"]


def test_snapshot_provider_models_remain_required_for_openrouter_embeddings(tmp_path):
    store = ProviderSnapshotStore(str(tmp_path), mode=MODE_RECORD)
    request = build_snapshot_request(
        "POST",
        "https://openrouter.ai/api/v1/embeddings",
        body={"model": "openai/text-embedding-production", "input": "probe"},
    )
    store.record_response(request, status=200, body_text="{}")

    assert store.provider_model_request_counts() == {"openrouter": 1}
    assert recorder._resolve_snapshot_provider_model_ids(
        store=store,
        observed=["openai/text-embedding-production"],
        declared=[],
    ) == ["openai/text-embedding-production"]


def test_snapshot_provider_models_are_optional_for_openrouter_control_requests(tmp_path):
    store = _store_with_openrouter_control_request(tmp_path)

    assert store.provider_request_counts() == {"openrouter": 1}
    assert store.provider_model_request_counts() == {}
    assert recorder._resolve_snapshot_provider_model_ids(
        store=store,
        observed=[],
        declared=[],
    ) == []


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/v1/keys"),
        ("PATCH", "/api/v1/workspaces/workspace-id"),
        ("DELETE", "/api/v1/keys/key-id"),
    ],
)
def test_snapshot_provider_models_allow_approved_openrouter_control_mutations(
    tmp_path,
    method,
    path,
):
    store = _store_with_openrouter_control_request(
        tmp_path,
        method=method,
        path=path,
    )

    assert store.provider_model_request_counts() == {}


def test_snapshot_provider_models_reject_unknown_openrouter_mutation(tmp_path):
    store = _store_with_openrouter_control_request(
        tmp_path,
        method="POST",
        path="/api/v1/unknown-operation",
    )

    with pytest.raises(
        DevSnapshotStoreError,
        match="neither model inference nor an approved control operation",
    ):
        store.provider_model_request_counts()

    store.write_dev_icp_items(
        [
            {
                "icp_ref": "test:1",
                "icp_hash": "sha256:" + "1" * 64,
                "icp": {"industry": "Software Development"},
            }
        ]
    )
    manifest = store.build_manifest(
        icp_set_hash="sha256:" + "2" * 64,
        provenance={"provider_model_ids": []},
    )
    verification = store.verify_manifest(manifest)
    assert verification["passed"] is False
    assert "snapshot_record_invalid:DevSnapshotStoreError" in verification["errors"]


def test_snapshot_provider_models_reject_unattributed_openrouter_rows(tmp_path):
    store = _store_with_provider_request(tmp_path, "exa")

    with pytest.raises(ValueError, match="has no model-bearing snapshot request"):
        recorder._resolve_snapshot_provider_model_ids(
            store=store,
            observed=["openai/gpt-production"],
            declared=["openai/gpt-production"],
        )


@pytest.mark.parametrize(
    ("provider", "provider_model_ids", "expected_error"),
    [
        ("exa", [], ""),
        ("openrouter", ["openai/gpt-production"], ""),
        (
            "openrouter",
            [],
            "snapshot_provenance_missing:provider_model_ids",
        ),
        (
            "exa",
            ["openai/gpt-production"],
            "snapshot_provenance_unattributed:provider_model_ids",
        ),
    ],
)
def test_manifest_binds_model_provenance_to_recorded_provider_requests(
    tmp_path,
    provider,
    provider_model_ids,
    expected_error,
):
    store = _store_with_provider_request(tmp_path, provider)
    store.write_dev_icp_items(
        [
            {
                "icp_ref": "test:1",
                "icp_hash": "sha256:" + "1" * 64,
                "icp": {"industry": "Software Development"},
            }
        ]
    )
    manifest = store.build_manifest(
        icp_set_hash="sha256:" + "2" * 64,
        provenance={"provider_model_ids": provider_model_ids},
    )

    verification = store.verify_manifest(manifest)

    assert verification["provider_request_counts"] == {provider: 1}
    assert verification["provider_model_request_counts"] == (
        {"openrouter": 1} if provider == "openrouter" else {}
    )
    if expected_error:
        assert verification["passed"] is False
        assert expected_error in verification["errors"]
    else:
        assert verification["passed"] is True, verification["errors"]


def test_manifest_allows_openrouter_control_traffic_without_model_provenance(tmp_path):
    store = _store_with_openrouter_control_request(tmp_path)
    store.write_dev_icp_items(
        [
            {
                "icp_ref": "test:1",
                "icp_hash": "sha256:" + "1" * 64,
                "icp": {"industry": "Software Development"},
            }
        ]
    )
    manifest = store.build_manifest(
        icp_set_hash="sha256:" + "2" * 64,
        provenance={"provider_model_ids": []},
    )

    verification = store.verify_manifest(manifest)

    assert verification["passed"] is True, verification["errors"]
    assert verification["provider_request_counts"] == {"openrouter": 1}
    assert verification["provider_model_request_counts"] == {}
