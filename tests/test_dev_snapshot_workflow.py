from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from gateway.research_lab import snapshot_refresh
from tests.restart_rehearsal.dev_snapshot_workflow import (
    BOUNDARY_SCHEMA,
    SCENARIO_INVARIANT,
    SCENARIO_NAME,
    _source_root,
    expected_cli_argv_contract_hashes,
    expected_docker_bootstrap_hashes,
    exercise_dev_snapshot_downstream_publication,
)


_PASS_PREDICATES = (
    "production_command_timeout_teardown_exact",
    "production_command_cancellation_teardown_exact",
    "production_commands_exact",
    "production_command_process_groups_isolated",
    "production_cli_argv_contracts_exact",
    "docker_bootstrap_contracts_exact",
    "configured_baseline_complete",
    "supabase_export_exact",
    "provider_record_replay_exact",
    "docker_daemon_readiness_exact",
    "immutable_ready_before_pointer",
    "signed_readiness_exact",
    "active_identity_rechecked",
    "immutable_target_exact",
    "cleanup_complete",
    "unknown_boundaries_rejected",
    "alternate_http_seams_fail_closed",
    "unknown_subprocess_rejected",
    "production_git_discovery_fail_closed",
    "unknown_aws_service_rejected",
    "declared_boundary_operations_exact",
    "negative_boundary_evidence_complete",
)


def test_exact_dev_snapshot_downstream_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    candidate = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    monkeypatch.setenv("REHEARSAL_SOURCE_ROOT", str(source_root))
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", candidate)
    result = exercise_dev_snapshot_downstream_publication()

    assert result["scenario"] == SCENARIO_NAME
    assert result["invariant"] == SCENARIO_INVARIANT
    assert all(result[name] is True for name in _PASS_PREDICATES)
    assert result["configured_bank_size"] >= result["configured_tree_width"] > 0
    assert result["container_execution_count"] == 3 * result["configured_bank_size"]
    assert (
        result["docker_daemon_readiness_count"]
        == result["container_execution_count"]
    )
    assert result["provider_request_count"] == 3 * result["configured_bank_size"]
    assert result["production_cli_argv_contract_hashes"] == list(
        expected_cli_argv_contract_hashes().values()
    )
    assert result["docker_bootstrap_contract_hashes"] == (
        expected_docker_bootstrap_hashes()
    )
    assert result["negative_probe_ids"] == {
        "requests": 1,
        "urllib": 1,
        "httpx_sync": 1,
        "httpx_async": 1,
        "aiohttp": 1,
        "subprocess": 1,
        "docker_argv": 1,
        "aws_service": 1,
    }
    assert result["production_git_rejection_count"] == 2


def _boundary_state(tmp_path: Path, source_root: Path) -> Path:
    champion_root = tmp_path / "champion"
    champion_root.mkdir(exist_ok=True)
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": BOUNDARY_SCHEMA,
                "root": str(tmp_path),
                "source_root": str(source_root),
                "champion_root": str(champion_root),
                "bucket": "rehearsal-dev-snapshots",
                "base_prefix": "research-lab/dev-snapshots/",
                "kms_key_id": "alias/rehearsal-dev-snapshot",
                "active_artifact": {
                    "image_digest": "rehearsal.invalid/champion@sha256:" + "a" * 64,
                    "git_commit_sha": "b" * 40,
                    "config_hash": "sha256:" + "c" * 64,
                    "manifest_hash": "sha256:" + "d" * 64,
                },
                "selection_seed": "exact-rehearsal-snapshot",
                "provider_model_ids": ["openai/rehearsal-model"],
                "expected_cli_argv_contract_hashes": (
                    expected_cli_argv_contract_hashes()
                ),
                "expected_docker_bootstrap_hashes": (
                    expected_docker_bootstrap_hashes()
                ),
                "supabase_tables": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return state_path


def _boundary_environment(state_path: Path, source_root: Path) -> dict[str, str]:
    boundary_root = (
        source_root / "tests" / "restart_rehearsal" / "dev_snapshot_boundary"
    )
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(boundary_root), str(source_root))),
        "REHEARSAL_DEV_SNAPSHOT_BOUNDARY_STATE": str(state_path),
    }


def test_source_root_is_explicit_and_flat_harness_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "gateway").mkdir(parents=True)
    (candidate / "scripts").mkdir()
    monkeypatch.setenv("REHEARSAL_SOURCE_ROOT", str(candidate))
    monkeypatch.setattr(
        "tests.restart_rehearsal.dev_snapshot_workflow.__file__",
        "/harness/dev_snapshot_workflow.py",
    )

    assert _source_root() == candidate.resolve()

    monkeypatch.setenv("REHEARSAL_SOURCE_ROOT", str(tmp_path / "missing"))
    with pytest.raises(
        RuntimeError,
        match="REHEARSAL_SOURCE_ROOT is not a candidate source tree",
    ):
        _source_root()


def test_dev_snapshot_boundary_rejects_unknown_provider(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    state_path = _boundary_state(tmp_path, source_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import requests; requests.get('https://undeclared.invalid/path')",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=_boundary_environment(state_path, source_root),
    )

    assert completed.returncode != 0
    assert "dev-snapshot HTTP seam requests is not allowlisted" in completed.stderr


def test_dev_snapshot_boundary_rejects_unknown_subprocess_and_bad_cli_argv(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    state_path = _boundary_state(tmp_path, source_root)
    env = _boundary_environment(state_path, source_root)

    unknown = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess; subprocess.run(['/dev-snapshot-undeclared'])",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=env,
    )
    export_script = source_root / "scripts" / "export_research_lab_dev_icp_inputs.py"
    bad_cli_source = (
        "import subprocess, sys\n"
        f"subprocess.run([sys.executable, {str(export_script)!r}, "
        "'--out-dir', '/tmp/escape'])\n"
    )
    bad_cli = subprocess.run(
        [sys.executable, "-c", bad_cli_source],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=env,
    )

    assert unknown.returncode != 0
    assert "dev-snapshot subprocess operation is not allowlisted" in unknown.stderr
    assert bad_cli.returncode != 0
    assert "dev-snapshot export argv shape differs" in bad_cli.stderr


def test_dev_snapshot_boundary_rejects_tampered_docker_bootstrap(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    state_path = _boundary_state(tmp_path, source_root)
    work_root = tmp_path / "work" / "refresh-999-negative"
    staging = work_root / (".snapshot.recording.999." + "e" * 32)
    staging.mkdir(parents=True)
    icp_ref = "qualification_private_icp_sets:negative"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        "leadpoet-dev-snapshot-record-" + "f" * 32,
        "-i",
        "-v",
        f"{staging}:/research_lab_dev_snapshots",
        "-e",
        "RESEARCH_LAB_DEV_SNAPSHOT_DIR=/research_lab_dev_snapshots",
        "-e",
        f"RESEARCH_LAB_DEV_RECORD_ICP_REF={icp_ref}",
        "rehearsal.invalid/champion@sha256:" + "a" * 64,
        "python",
        "-c",
        "tampered-bootstrap",
        "research_lab_adapter",
        "run_icp",
    ]
    payload = {
        "icp": {"icp_id": "dev-snapshot-001"},
        "context": {
            "dev_snapshot_recording": True,
            "runtime_options": {},
        },
    }
    source = (
        "import json, os, subprocess\n"
        f"command = {command!r}\n"
        f"payload = {payload!r}\n"
        f"icp_ref = {icp_ref!r}\n"
        "subprocess.run(command, input=json.dumps(payload), text=True, "
        "capture_output=True, timeout=900, "
        "env={'PATH': os.environ.get('PATH', ''), "
        "'RESEARCH_LAB_DEV_RECORD_ICP_REF': icp_ref}, check=False)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=_boundary_environment(state_path, source_root),
    )

    assert completed.returncode != 0
    assert "dev-snapshot Docker argv or environment differs" in completed.stderr
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event.get("kind") == "subprocess"
        and event.get("operation") == "rejected"
        and event.get("command_class") == "docker"
        for event in events
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_snapshot_timeout_executes_production_group_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = tmp_path / "grandchild-started"
    escaped = tmp_path / "grandchild-escaped"
    grandchild = (
        "import pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(1.5)\n"
        f"pathlib.Path({str(escaped)!r}).write_text('escaped', encoding='utf-8')\n"
    )
    leader = (
        "import pathlib, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"child = subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        f"pathlib.Path({str(started)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n"
    )

    monkeypatch.setattr(snapshot_refresh, "COMMAND_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        snapshot_refresh,
        "COMMAND_TERMINATION_GRACE_SECONDS",
        0.1,
    )
    monkeypatch.setattr(snapshot_refresh, "COMMAND_KILL_WAIT_SECONDS", 3.0)
    with pytest.raises(subprocess.TimeoutExpired):
        snapshot_refresh._run_command(
            [sys.executable, "-c", leader],
            os.environ,
            0.5,
        )

    assert started.is_file()
    grandchild_pid = int(started.read_text(encoding="utf-8"))
    assert not _pid_alive(grandchild_pid)
    time.sleep(1.6)
    assert not escaped.exists()


def test_snapshot_cancellation_executes_production_group_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = tmp_path / "cancelled-started"
    command_source = (
        "import os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(started)!r}).write_text(str(os.getpid()))\n"
        "os.close(1)\n"
        "os.close(2)\n"
        "while True: time.sleep(1)\n"
    )
    monkeypatch.setattr(snapshot_refresh, "COMMAND_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        snapshot_refresh,
        "COMMAND_TERMINATION_GRACE_SECONDS",
        0.5,
    )
    monkeypatch.setattr(snapshot_refresh, "COMMAND_KILL_WAIT_SECONDS", 3.0)

    async def scenario() -> None:
        command = asyncio.create_task(
            snapshot_refresh._await_command_completion(
                snapshot_refresh._run_command,
                [sys.executable, "-c", command_source],
                os.environ,
                60,
            )
        )
        deadline = asyncio.get_running_loop().time() + 5
        while not started.is_file():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("production cancellation process did not start")
            await asyncio.sleep(0.01)
        pid = int(started.read_text(encoding="utf-8"))
        command.cancel()
        await asyncio.sleep(0.1)
        assert not command.done()
        assert _pid_alive(pid)
        with pytest.raises(asyncio.CancelledError):
            await command
        assert not _pid_alive(pid)

    asyncio.run(scenario())
