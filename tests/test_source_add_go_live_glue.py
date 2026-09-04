"""Go-live glue: docker trial runner and probe-guard wiring."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from gateway.research_lab.source_add_trial_runner import (
    build_source_add_sandbox_runner,
    build_trial_registry_entry,
)
from research_lab.source_add_execution import (
    SourceAddRejectionReason,
    intake_source_add_submission,
    run_sandboxed_trial,
)


def _manifest_doc(**overrides: Any) -> dict[str, Any]:
    doc = {
        "adapter_id": "adapter:glue-test-1",
        "miner_ref": "miner:hotkey-g",
        "source_name": "Glue Test Source",
        "source_kind": "news",
        "declared_base_domains": ["gluefeed.example"],
        "output_schema_ref": "schema:source-add-output:v1",
        "allowed_output_fields": ["evidence_refs", "snapshot_refs", "content_hashes", "normalized_text_hashes"],
        "submitted_artifact_ref": "artifact:glue",
        "code_bundle_hash": "sha256:" + "a" * 64,
        "sandbox_policy_ref": "policy:sandbox-v1",
        "max_trial_cost_cents": 500,
        "max_request_cost_cents": 5,
        "max_latency_ms": 30_000,
        "fixture_refs": ["fixture:glue"],
    }
    doc.update(overrides)
    return doc


def _submission(**overrides: Any):
    record, errors = intake_source_add_submission(_manifest_doc(**overrides), miner_hotkey="hk-glue")
    assert errors == []
    return record


def _output_doc(adapter_id: str, icp_ref: str) -> dict[str, Any]:
    return {
        "output_ref": f"output:{icp_ref}",
        "adapter_id": adapter_id,
        "icp_ref": icp_ref,
        "evidence_refs": [f"evidence:{icp_ref}:0"],
        "snapshot_refs": [f"snapshot:{icp_ref}"],
        "content_hashes": ["sha256:" + "b" * 64],
        "normalized_text_hashes": ["sha256:" + "c" * 64],
    }


class TestTrialRegistryEntry:
    def test_entry_derives_from_manifest(self):
        record = _submission()
        entry = build_trial_registry_entry(record)
        assert entry.base_url == "https://gluefeed.example"
        assert entry.auth_kind == "none"  # no_credentials manifest
        assert entry.id.startswith("trial_")

    def test_legacy_miner_credential_manifest_is_rejected_at_intake(self):
        record, errors = intake_source_add_submission(
            _manifest_doc(
            adapter_id="adapter:glue-cred-1",
            credential_policy="credential_ref_only",
            credential_ref="encrypted_ref:source_add:abc",
            ),
            miner_hotkey="hk-glue",
        )
        assert record is None
        assert any(
            SourceAddRejectionReason.CREDENTIAL_INVALID.value in item
            for item in errors
        )


class TestSandboxRunner:
    def _run(self, tmp_path: Path, docker_exec, record=None, miner_credential=""):
        record = record or _submission()
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "adapter.py").write_text("print('{}')\n", encoding="utf-8")
        runner, shutdown = build_source_add_sandbox_runner(
            record=record,
            bundle_dir=bundle,
            work_dir=tmp_path / "work",
            miner_credential=miner_credential,
            docker_exec=docker_exec,
        )
        try:
            return record, runner(record, "icp:1")
        finally:
            shutdown()

    def test_successful_run_parses_last_json_line(self, tmp_path):
        record = _submission()
        output = _output_doc(record.adapter_id, "icp:1")

        def _fake_docker(argv, timeout):
            assert argv[:3] == ["docker", "run", "--rm"]
            assert any("RESEARCH_LAB_EVIDENCE_PROXY_URL=" in arg for arg in argv)
            return 0, "progress line\n" + json.dumps(output) + "\n", ""

        _, result = self._run(tmp_path, _fake_docker, record=record)
        assert result["output"] == output
        assert result["cost_cents"] == 0

    def test_nonzero_exit_maps_to_error(self, tmp_path):
        _, result = self._run(tmp_path, lambda argv, timeout: (3, "", "boom"))
        assert result["error"] == "adapter_exit_3"

    def test_missing_output_json_maps_to_error(self, tmp_path):
        _, result = self._run(tmp_path, lambda argv, timeout: (0, "no json here", ""))
        assert result["error"] == "no_output_json"

    def test_timeout_maps_to_timeout_error(self, tmp_path):
        import subprocess

        def _hang(argv, timeout):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

        _, result = self._run(tmp_path, _hang)
        assert result["error"] == "timeout"

    def test_bundle_without_entrypoint_rejected(self, tmp_path):
        record = _submission()
        empty_bundle = tmp_path / "empty"
        empty_bundle.mkdir()
        with pytest.raises(ValueError, match="adapter.py"):
            build_source_add_sandbox_runner(
                record=record,
                bundle_dir=empty_bundle,
                work_dir=tmp_path / "work",
            )

    def test_legacy_trial_has_no_credential_for_public_submission(self):
        record = _submission(adapter_id="adapter:glue-proxy-1")
        entry = build_trial_registry_entry(record)
        assert entry.auth_kind == "none"
        assert entry.credential_ref == ()

    def test_default_executor_holds_shared_docker_lifecycle(
        self,
        tmp_path,
        monkeypatch,
    ):
        from gateway.research_lab import source_add_trial_runner as runner_module
        from research_lab import docker_operation_lock_v2 as docker_lock

        record = _submission(adapter_id="adapter:glue-lock-1")
        output = _output_doc(record.adapter_id, "icp:1")
        lock_file = tmp_path / "docker-operation.lock"
        monkeypatch.setenv(
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE", str(lock_file)
        )
        monkeypatch.setattr(
            docker_lock.subprocess,
            "run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                args=["docker", "info"], returncode=0
            ),
        )

        def _fake_default(argv, timeout):
            assert argv[:3] == ["docker", "run", "--rm"]
            assert timeout > 0
            exclusive_fd = os.open(
                lock_file, os.O_CREAT | os.O_RDWR, 0o600
            )
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(
                        exclusive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
            finally:
                os.close(exclusive_fd)
            return 0, json.dumps(output), ""

        monkeypatch.setattr(
            runner_module, "_subprocess_docker_exec", _fake_default
        )
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "adapter.py").write_text("print('{}')\n", encoding="utf-8")
        runner, shutdown = build_source_add_sandbox_runner(
            record=record,
            bundle_dir=bundle,
            work_dir=tmp_path / "work",
        )
        try:
            result = runner(record, "icp:1")
        finally:
            shutdown()
        assert result["output"] == output

    def test_default_executor_removes_named_container_on_timeout(
        self,
        tmp_path,
        monkeypatch,
    ):
        from gateway.research_lab import source_add_trial_runner as runner_module

        record = _submission(adapter_id="adapter:glue-timeout-1")
        lock_file = tmp_path / "docker-operation.lock"
        removed: list[str] = []
        monkeypatch.setenv(
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE", str(lock_file)
        )

        def _fake_run(command, **kwargs):
            command = list(command)
            if command[-1:] == ["info"]:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if command[1:3] == ["rm", "-f"]:
                exclusive_fd = os.open(
                    lock_file, os.O_CREAT | os.O_RDWR, 0o600
                )
                try:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(
                            exclusive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                finally:
                    os.close(exclusive_fd)
                removed.append(command[-1])
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

        monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "adapter.py").write_text("print('{}')\n", encoding="utf-8")
        runner, shutdown = build_source_add_sandbox_runner(
            record=record,
            bundle_dir=bundle,
            work_dir=tmp_path / "work",
            timeout_seconds=30,
        )
        try:
            result = runner(record, "icp:1")
        finally:
            shutdown()
        assert result["error"] == "timeout"
        assert len(removed) == 1
        assert removed[0].startswith("leadpoet-source-add-trial-")


class TestFullTrialThroughRunner:
    def test_run_sandboxed_trial_with_docker_runner_and_cost_metering(self, tmp_path):
        record = _submission(adapter_id="adapter:glue-cost-1")
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "adapter.py").write_text("print('{}')\n", encoding="utf-8")

        calls: list[str] = []

        def _fake_docker(argv, timeout):
            icp_ref = next(arg for arg in argv if arg.startswith("SOURCE_ADD_ICP_REF=")).split("=", 1)[1]
            calls.append(icp_ref)
            return 0, json.dumps(_output_doc(record.adapter_id, icp_ref)), ""

        runner, shutdown = build_source_add_sandbox_runner(
            record=record,
            bundle_dir=bundle,
            work_dir=tmp_path / "work",
            docker_exec=_fake_docker,
        )
        try:
            trial = run_sandboxed_trial(
                record,
                trial_icp_refs=("icp:1", "icp:2"),
                sandbox_runner=runner,
                evidence_classifier=lambda _ref: "news",
            )
        finally:
            shutdown()
        assert trial.failure_reason == ""
        assert trial.measured_trial_yield == 1.0
        assert calls == ["icp:1", "icp:2"]
