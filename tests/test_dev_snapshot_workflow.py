from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from gateway.research_lab import snapshot_refresh
from tests.restart_rehearsal.dev_snapshot_workflow import (
    BOUNDARY_SCHEMA,
    DevSnapshotWorkflowTimeout,
    SCENARIO_INVARIANT,
    SCENARIO_NAME,
    _PROCESS_GROUP_REGISTRY_DIR,
    _PROCESS_GROUP_REGISTRY_SCHEMA,
    _PROCESS_GROUP_SPAWN_DIR,
    _cleanup_registered_process_groups,
    _run_in_new_process_group,
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
    "production_command_spawn_gates_exact",
    "process_group_registry_cleanup_exact",
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
        "popen": 1,
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
    unknown_popen = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess; subprocess.Popen(['/dev-snapshot-undeclared-popen'])",
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
    assert unknown_popen.returncode != 0
    assert "dev-snapshot Popen operation is not allowlisted" in unknown_popen.stderr
    assert bad_cli.returncode != 0
    assert "dev-snapshot export argv shape differs" in bad_cli.stderr


def test_dev_snapshot_boundary_rejects_popen_execution_and_env_overrides(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    state_path = _boundary_state(tmp_path, source_root)
    env = _boundary_environment(state_path, source_root)
    inputs_dir = tmp_path / "work" / "refresh-999-popen-contract" / "inputs"
    export_script = source_root / "scripts" / "export_research_lab_dev_icp_inputs.py"
    production_command = [
        sys.executable,
        str(export_script),
        "--out-dir",
        str(inputs_dir),
        "--seed",
        "exact-rehearsal-snapshot",
        "--expected-private-model-manifest-hash",
        "sha256:" + "d" * 64,
    ]
    common = (
        f"command = {production_command!r}\n"
        "options = dict(text=True, stdout=subprocess.PIPE, "
        "stderr=subprocess.PIPE, start_new_session=True)\n"
    )
    executable_override = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, subprocess\n"
            + common
            + "subprocess.Popen(command, env=dict(os.environ), "
            "executable='/bin/true', **options)\n",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=env,
    )
    stripped_boundary_env = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, subprocess\n"
            + common
            + "child_env = dict(os.environ)\n"
            + "child_env.pop('REHEARSAL_DEV_SNAPSHOT_BOUNDARY_STATE')\n"
            + "subprocess.Popen(command, env=child_env, **options)\n",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=env,
    )
    run_override = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, subprocess\n"
            + common
            + "subprocess.run(command, text=True, capture_output=True, "
            + "env=dict(os.environ), executable='/bin/true')\n",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=env,
    )

    for completed in (executable_override, stripped_boundary_env):
        assert completed.returncode != 0
        assert (
            "dev-snapshot production process-group contract differs"
            in completed.stderr
        )
    assert run_override.returncode != 0
    assert (
        "dev-snapshot production commands require the private Popen contract"
        in run_override.stderr
    )
    assert not inputs_dir.exists()


def test_dev_snapshot_boundary_real_run_bypass_is_thread_local(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    state_path = _boundary_state(tmp_path, source_root)
    source = (
        "import sitecustomize, subprocess, threading\n"
        "entered = threading.Event()\n"
        "release = threading.Event()\n"
        "def fake_run(*_args, **_kwargs):\n"
        "    entered.set()\n"
        "    assert release.wait(5)\n"
        "    return subprocess.CompletedProcess([], 0)\n"
        "sitecustomize._real_subprocess_run = fake_run\n"
        "owner = threading.Thread(target=lambda: "
        "sitecustomize._run_validated_real_subprocess(['internal']))\n"
        "owner.start()\n"
        "assert entered.wait(5)\n"
        "try:\n"
        "    subprocess.Popen(['/bin/true'])\n"
        "except RuntimeError as exc:\n"
        "    assert str(exc) == 'dev-snapshot Popen operation is not allowlisted'\n"
        "else:\n"
        "    raise RuntimeError('cross-thread Popen escaped the strict boundary')\n"
        "finally:\n"
        "    release.set()\n"
        "    owner.join(5)\n"
        "assert not owner.is_alive()\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
        env=_boundary_environment(state_path, source_root),
    )

    assert completed.returncode == 0, completed.stderr


def test_outer_timeout_resolves_spawn_marker_before_controller_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    state_path = _boundary_state(tmp_path, source_root)
    work_root = tmp_path / "work" / "refresh-999-spawn-race"
    inputs_dir = work_root / "inputs"
    started = tmp_path / "nested-started"
    export_script = source_root / "scripts" / "export_research_lab_dev_icp_inputs.py"
    production_arguments = [
        str(export_script),
        "--out-dir",
        str(inputs_dir),
        "--seed",
        "exact-rehearsal-snapshot",
        "--expected-private-model-manifest-hash",
        "sha256:" + "d" * 64,
    ]
    source = (
        "import sitecustomize, subprocess, sys, time\n"
        "real_popen = sitecustomize._real_subprocess_popen\n"
        "def paused_popen(*args, **kwargs):\n"
        "    child = real_popen(*args, **kwargs)\n"
        f"    __import__('pathlib').Path({str(started)!r}).write_text(str(child.pid))\n"
        "    time.sleep(1.5)\n"
        "    return child\n"
        "sitecustomize._real_subprocess_popen = paused_popen\n"
        f"process = subprocess.Popen([sys.executable, *{production_arguments!r}], text=True, "
        "stdout=subprocess.PIPE, stderr=subprocess.PIPE, "
        "start_new_session=True, env=dict(__import__('os').environ))\n"
        "process.communicate()\n"
    )
    monkeypatch.setattr(
        "tests.restart_rehearsal.dev_snapshot_workflow."
        "_PROCESS_GROUP_SPAWN_RESOLUTION_SECONDS",
        0.1,
    )

    try:
        completed = _run_in_new_process_group(
            [sys.executable, "-c", source],
            env=_boundary_environment(state_path, source_root),
            timeout_seconds=1.0,
            label="forced spawn-marker race",
            term_grace_seconds=0.1,
        )
    except DevSnapshotWorkflowTimeout:
        pass
    else:
        pytest.fail(
            "forced spawn-marker race exited before timeout: "
            + str(completed.stderr or completed.stdout or "")[-1200:]
        )

    assert started.is_file()
    nested_pid = int(started.read_text(encoding="utf-8"))
    assert not _pid_alive(nested_pid)
    assert not (inputs_dir / "source_icps.json").exists()
    for directory_name in (
        _PROCESS_GROUP_REGISTRY_DIR,
        _PROCESS_GROUP_SPAWN_DIR,
    ):
        directory = tmp_path / directory_name
        assert not directory.exists() or not any(directory.iterdir())


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


def test_outer_success_reaps_same_group_descendant_that_closed_pipes(
    tmp_path: Path,
) -> None:
    started = tmp_path / "closed-pipe-descendant.pid"
    escaped = tmp_path / "closed-pipe-descendant-escaped"
    descendant = (
        "import os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "os.close(1)\n"
        "os.close(2)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(1.5)\n"
        "pathlib.Path(sys.argv[2]).write_text('escaped')\n"
        "while True: time.sleep(1)\n"
    )
    leader = (
        "import pathlib, subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[3], sys.argv[1], sys.argv[2]])\n"
        "deadline = time.monotonic() + 2\n"
        "while not pathlib.Path(sys.argv[1]).is_file():\n"
        "    if time.monotonic() >= deadline: raise SystemExit(2)\n"
        "    time.sleep(0.01)\n"
    )

    with pytest.raises(
        RuntimeError,
        match="child exited with a live process-group descendant",
    ):
        _run_in_new_process_group(
            [
                sys.executable,
                "-c",
                leader,
                str(started),
                str(escaped),
                descendant,
            ],
            env=os.environ,
            timeout_seconds=5,
            label="closed-pipe descendant",
            term_grace_seconds=0.1,
        )

    descendant_pid = int(started.read_text(encoding="utf-8"))
    assert not _pid_alive(descendant_pid)
    time.sleep(1.6)
    assert not escaped.exists()


def test_registered_pgid_survives_reaped_leader_until_descendant_cleanup(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    state_path = _boundary_state(tmp_path, source_root)
    env = _boundary_environment(state_path, source_root)
    started = tmp_path / "registered-descendant.pid"
    descendant = (
        "import os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    leader = (
        "import pathlib, subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]])\n"
        "deadline = time.monotonic() + 2\n"
        "while not pathlib.Path(sys.argv[1]).is_file():\n"
        "    if time.monotonic() >= deadline: raise SystemExit(2)\n"
        "    time.sleep(0.01)\n"
    )
    outer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    nested = subprocess.Popen(
        [sys.executable, "-c", leader, str(started), descendant],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert nested.wait(timeout=5) == 0
        descendant_pid = int(started.read_text(encoding="utf-8"))
        registry = tmp_path / _PROCESS_GROUP_REGISTRY_DIR
        registry.mkdir()
        (registry / f"{nested.pid}.json").write_text(
            json.dumps(
                {
                    "schema_version": _PROCESS_GROUP_REGISTRY_SCHEMA,
                    "status": "active",
                    "pid": nested.pid,
                    "pgid": nested.pid,
                    "owner_pid": outer.pid,
                    "owner_pgid": outer.pid,
                    "spawn_nonce": "a" * 32,
                    "phase": "export",
                    "command_name": "export_research_lab_dev_icp_inputs.py",
                    "python_executable": sys.executable,
                    "script_path": str(
                        source_root
                        / "scripts"
                        / "export_research_lab_dev_icp_inputs.py"
                    ),
                    "argv_contract_hash": expected_cli_argv_contract_hashes()[
                        "export"
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        _cleanup_registered_process_groups(
            outer,
            env=env,
            term_grace_seconds=0.1,
        )

        assert not _pid_alive(descendant_pid)
        assert not any(registry.iterdir())
    finally:
        try:
            os.killpg(outer.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        outer.wait(timeout=5)
        try:
            os.killpg(nested.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_registered_successful_zombie_group_is_pruned_on_darwin_eperm(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    state_path = _boundary_state(tmp_path, source_root)
    env = _boundary_environment(state_path, source_root)
    started = tmp_path / "zombie-leader.pid"
    owner_source = (
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'pass'], "
        "start_new_session=True)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
        "while True: time.sleep(1)\n"
    )
    outer = subprocess.Popen(
        [sys.executable, "-c", owner_source, str(started)],
        start_new_session=True,
    )
    real_killpg = os.killpg
    try:
        start_deadline = time.monotonic() + 5
        while not started.is_file():
            if time.monotonic() >= start_deadline:
                pytest.fail("zombie-group owner did not publish its child PID")
            time.sleep(0.01)
        child_pid = int(started.read_text(encoding="utf-8"))
        zombie_deadline = time.monotonic() + 5
        while True:
            state = subprocess.run(
                ["ps", "-o", "state=", "-p", str(child_pid)],
                text=True,
                capture_output=True,
                check=False,
                timeout=2,
            ).stdout.strip()
            if state.startswith("Z"):
                break
            if time.monotonic() >= zombie_deadline:
                pytest.fail(f"registered child did not become a zombie: {state!r}")
            time.sleep(0.01)

        registry = tmp_path / _PROCESS_GROUP_REGISTRY_DIR
        registry.mkdir()
        row_path = registry / f"{child_pid}.json"
        row_path.write_text(
            json.dumps(
                {
                    "schema_version": _PROCESS_GROUP_REGISTRY_SCHEMA,
                    "status": "active",
                    "pid": child_pid,
                    "pgid": child_pid,
                    "owner_pid": outer.pid,
                    "owner_pgid": outer.pid,
                    "spawn_nonce": "c" * 32,
                    "phase": "export",
                    "command_name": "export_research_lab_dev_icp_inputs.py",
                    "python_executable": sys.executable,
                    "script_path": str(
                        source_root
                        / "scripts"
                        / "export_research_lab_dev_icp_inputs.py"
                    ),
                    "argv_contract_hash": expected_cli_argv_contract_hashes()[
                        "export"
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        zombie_probe_permission_denied = False
        try:
            real_killpg(child_pid, 0)
        except PermissionError:
            zombie_probe_permission_denied = True
        if sys.platform == "darwin":
            assert zombie_probe_permission_denied

        assert _cleanup_registered_process_groups(
            outer,
            env=env,
            term_grace_seconds=0.1,
        )
        assert not row_path.exists()
        outer.wait(timeout=5)
    finally:
        if outer.poll() is None:
            try:
                real_killpg(outer.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            outer.wait(timeout=5)


def test_registered_process_group_inspection_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.restart_rehearsal import dev_snapshot_workflow

    process_group_id = 43210

    def permission_denied(_process_group_id: int, _signum: int) -> None:
        raise PermissionError

    monkeypatch.setattr(dev_snapshot_workflow.os, "killpg", permission_denied)

    def completed(
        stdout: str,
        *,
        returncode: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["ps"],
            returncode,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(
        dev_snapshot_workflow.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(
            f"{process_group_id} 1 {process_group_id} S\n"
        ),
    )
    assert dev_snapshot_workflow._registered_process_group_id_alive(
        process_group_id
    )

    monkeypatch.setattr(
        dev_snapshot_workflow.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(""),
    )
    assert dev_snapshot_workflow._registered_process_group_id_alive(
        process_group_id
    )

    monkeypatch.setattr(
        dev_snapshot_workflow.subprocess,
        "run",
        lambda *_args, **_kwargs: completed("ambiguous\n"),
    )
    assert dev_snapshot_workflow._registered_process_group_id_alive(
        process_group_id
    )

    monkeypatch.setattr(
        dev_snapshot_workflow.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(
            f"{process_group_id} 1 {process_group_id} Z\n"
        ),
    )
    assert not dev_snapshot_workflow._registered_process_group_id_alive(
        process_group_id
    )


@pytest.mark.parametrize(
    ("termination_error", "expected_target_probes"),
    (
        (ProcessLookupError, 1),
        (PermissionError, 2),
    ),
)
def test_registered_pgid_exit_between_probe_and_term_is_pruned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination_error: type[BaseException],
    expected_target_probes: int,
) -> None:
    from tests.restart_rehearsal import dev_snapshot_workflow

    source_root = Path(__file__).resolve().parents[1]
    state_path = _boundary_state(tmp_path, source_root)
    env = _boundary_environment(state_path, source_root)
    controller_ready = tmp_path / "controller-ready"
    advance_trigger = tmp_path / "advance-trigger"
    advanced = tmp_path / "advanced"
    outer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, sys, time\n"
                "pathlib.Path(sys.argv[1]).write_text('ready')\n"
                "trigger = pathlib.Path(sys.argv[2])\n"
                "while not trigger.exists(): time.sleep(0.001)\n"
                "pathlib.Path(sys.argv[3]).write_text('advanced')\n"
                "while True: time.sleep(1)\n"
            ),
            str(controller_ready),
            str(advance_trigger),
            str(advanced),
        ],
        start_new_session=True,
    )
    finished = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        start_new_session=True,
    )
    real_killpg = os.killpg
    try:
        readiness_deadline = time.monotonic() + 5
        while not controller_ready.is_file():
            if time.monotonic() >= readiness_deadline:
                pytest.fail("outer controller did not become ready")
            time.sleep(0.01)
        assert finished.wait(timeout=5) == 0
        registry = tmp_path / _PROCESS_GROUP_REGISTRY_DIR
        registry.mkdir()
        row_path = registry / f"{finished.pid}.json"
        row_path.write_text(
            json.dumps(
                {
                    "schema_version": _PROCESS_GROUP_REGISTRY_SCHEMA,
                    "status": "active",
                    "pid": finished.pid,
                    "pgid": finished.pid,
                    "owner_pid": outer.pid,
                    "owner_pgid": outer.pid,
                    "spawn_nonce": "b" * 32,
                    "phase": "export",
                    "command_name": "export_research_lab_dev_icp_inputs.py",
                    "python_executable": sys.executable,
                    "script_path": str(
                        source_root
                        / "scripts"
                        / "export_research_lab_dev_icp_inputs.py"
                    ),
                    "argv_contract_hash": expected_cli_argv_contract_hashes()[
                        "export"
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        target_probes = 0
        real_group_alive = (
            dev_snapshot_workflow._registered_process_group_id_alive
        )

        def raced_group_alive(process_group_id: int) -> bool:
            nonlocal target_probes
            if process_group_id == finished.pid:
                target_probes += 1
                return target_probes == 1
            return real_group_alive(process_group_id)

        def raced_killpg(process_group_id: int, signum: int) -> None:
            if process_group_id == finished.pid and signum == signal.SIGTERM:
                advance_trigger.write_text("advance", encoding="utf-8")
                raise termination_error
            real_killpg(process_group_id, signum)

        monkeypatch.setattr(
            dev_snapshot_workflow,
            "_registered_process_group_id_alive",
            raced_group_alive,
        )
        monkeypatch.setattr(dev_snapshot_workflow.os, "killpg", raced_killpg)

        assert _cleanup_registered_process_groups(
            outer,
            env=env,
            term_grace_seconds=0.1,
        )
        assert target_probes == expected_target_probes
        assert not row_path.exists()
        assert advance_trigger.is_file()
        time.sleep(0.1)
        assert not advanced.exists()
        outer.wait(timeout=5)
        assert not advanced.exists()
    finally:
        if outer.poll() is None:
            try:
                real_killpg(outer.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            outer.wait(timeout=5)


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
