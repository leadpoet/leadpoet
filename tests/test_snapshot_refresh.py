"""Restart, concurrency, and partial-failure tests for snapshot refresh."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from gateway.research_lab import snapshot_refresh
from gateway.research_lab.config import RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD
from gateway.research_lab.git_tree_models import TreePolicy
from research_lab.eval.snapshot_store import SNAPSHOT_URI_ENV


IMAGE = "123456789.dkr.ecr.test/model@sha256:" + "a" * 64
COMMIT = "b" * 40
CONFIG_HASH = "sha256:" + "c" * 64
MODEL_MANIFEST_HASH = "sha256:" + "f" * 64


def _active(
    image: str = IMAGE,
    commit: str = COMMIT,
    config_hash: str = CONFIG_HASH,
    manifest_hash: str = MODEL_MANIFEST_HASH,
):
    return SimpleNamespace(
        artifact=SimpleNamespace(
            image_digest=image,
            git_commit_sha=commit,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
        )
    )


def _ready(**overrides: Any) -> dict[str, Any]:
    return {
        "ready": True,
        "reason": "ready",
        "manifest_hash": "sha256:" + "d" * 64,
        "snapshot_age_seconds": 60,
        "champion_image_digest": IMAGE,
        "source_commit": COMMIT,
        "model_config_hash": CONFIG_HASH,
        "private_model_manifest_hash": MODEL_MANIFEST_HASH,
        **overrides,
    }


def _publish_output() -> str:
    return "snapshot_uri=s3://private-bucket/dev/" + "e" * 64 + "\n"


def _pipeline_output(command: Sequence[str], *, bank_size: int = 40) -> str:
    if command[1].endswith("export_research_lab_dev_icp_inputs.py"):
        out_dir = Path(command[command.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "source_icps.json").write_text(
            json.dumps(
                {
                    "schema_version": "research_lab.dev_icp_export.v2",
                    "items": [
                        {"icp_ref": f"test-icp-{index}"}
                        for index in range(bank_size)
                    ],
                    "daily_bank_manifest": {"bank_size": bank_size},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return f"daily_bank_icps={bank_size}\n"
    return _publish_output() if "--skip-current-pointer" in command else "ok"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_run_command_timeout_reaps_its_process_group(monkeypatch, tmp_path):
    pid_path = tmp_path / "timed-out.pid"
    monkeypatch.setattr(snapshot_refresh, "COMMAND_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        snapshot_refresh,
        "COMMAND_TERMINATION_GRACE_SECONDS",
        0.1,
    )
    monkeypatch.setattr(snapshot_refresh, "COMMAND_KILL_WAIT_SECONDS", 2.0)
    command = [
        sys.executable,
        "-c",
        (
            "import os, pathlib, sys, time\n"
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
            "while True: time.sleep(1)\n"
        ),
        str(pid_path),
    ]

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        snapshot_refresh._run_command(command, dict(os.environ), 2.0)

    assert time.monotonic() - started < 5.0
    pid = int(pid_path.read_text(encoding="utf-8"))
    assert not _pid_alive(pid)


def test_run_command_kills_term_ignoring_descendant_that_closed_pipes(
    monkeypatch, tmp_path
):
    descendant_pid_path = tmp_path / "descendant.pid"
    monkeypatch.setattr(snapshot_refresh, "COMMAND_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        snapshot_refresh,
        "COMMAND_TERMINATION_GRACE_SECONDS",
        0.1,
    )
    monkeypatch.setattr(snapshot_refresh, "COMMAND_KILL_WAIT_SECONDS", 3.0)
    descendant_code = (
        "import os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "os.close(1)\n"
        "os.close(2)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    launcher_code = (
        "import pathlib, subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]])\n"
        "deadline = time.monotonic() + 2\n"
        "while not pathlib.Path(sys.argv[1]).is_file():\n"
        "    if time.monotonic() >= deadline: raise SystemExit(2)\n"
        "    time.sleep(0.01)\n"
    )

    with pytest.raises(RuntimeError, match="left a live descendant process"):
        snapshot_refresh._run_command(
            [
                sys.executable,
                "-c",
                launcher_code,
                str(descendant_pid_path),
                descendant_code,
            ],
            dict(os.environ),
            2,
        )

    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    assert not _pid_alive(descendant_pid)


def test_run_command_immediately_reaps_descendant_that_retained_pipes(
    monkeypatch, tmp_path
):
    descendant_pid_path = tmp_path / "pipe-holder.pid"
    monkeypatch.setattr(snapshot_refresh, "COMMAND_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        snapshot_refresh,
        "COMMAND_TERMINATION_GRACE_SECONDS",
        0.1,
    )
    monkeypatch.setattr(snapshot_refresh, "COMMAND_KILL_WAIT_SECONDS", 3.0)
    descendant_code = (
        "import os, pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    launcher_code = (
        "import pathlib, subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]])\n"
        "deadline = time.monotonic() + 2\n"
        "while not pathlib.Path(sys.argv[1]).is_file():\n"
        "    if time.monotonic() >= deadline: raise SystemExit(2)\n"
        "    time.sleep(0.01)\n"
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="left a live descendant process"):
        snapshot_refresh._run_command(
            [
                sys.executable,
                "-c",
                launcher_code,
                str(descendant_pid_path),
                descendant_code,
            ],
            dict(os.environ),
            60,
        )

    assert time.monotonic() - started < 3.0
    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    assert not _pid_alive(descendant_pid)


def test_run_command_confirms_transient_group_exit_after_communicate(
    monkeypatch,
):
    observations = iter((True, False))
    observed_calls = 0

    def transient_group(_process_group_id: int) -> bool:
        nonlocal observed_calls
        observed_calls += 1
        return next(observations)

    monkeypatch.setattr(snapshot_refresh, "_process_group_alive", transient_group)

    assert snapshot_refresh._run_command(
        [sys.executable, "-c", "print('completed')"],
        dict(os.environ),
        2,
    ) == "completed\n"
    assert observed_calls == 2


def test_run_command_confirms_transient_group_exit_after_pipe_poll_timeout(
    monkeypatch,
):
    class CompletedProcessWithDelayedPipeDrain:
        pid = 424242
        returncode = None
        communicate_calls = 0

        def communicate(self, *, timeout):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(["probe"], timeout)
            self.returncode = 0
            return "completed\n", ""

        def poll(self):
            self.returncode = 0
            return self.returncode

    process = CompletedProcessWithDelayedPipeDrain()
    observations = iter((True, False, False))
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        snapshot_refresh,
        "_process_group_alive",
        lambda _process_group_id: next(observations),
    )
    monkeypatch.setattr(
        snapshot_refresh,
        "_terminate_process_group",
        lambda _process: pytest.fail("transient group was terminated"),
    )

    assert snapshot_refresh._run_command(
        [sys.executable, "-c", "print('completed')"],
        dict(os.environ),
        2,
    ) == "completed\n"
    assert process.communicate_calls == 2


def test_run_command_bounds_pipe_drain_after_completed_group_exit(monkeypatch):
    class CompletedProcessWithRetainedPipe:
        pid = 424242
        returncode = 0

        def communicate(self, *, timeout):
            raise subprocess.TimeoutExpired(["probe"], timeout)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: CompletedProcessWithRetainedPipe(),
    )
    monkeypatch.setattr(
        snapshot_refresh,
        "_process_group_alive",
        lambda _process_group_id: False,
    )
    monkeypatch.setattr(
        snapshot_refresh,
        "COMMAND_PIPE_DRAIN_TIMEOUT_SECONDS",
        0.0,
    )

    with pytest.raises(RuntimeError, match="pipes remained open"):
        snapshot_refresh._run_command(
            [sys.executable, "-c", "pass"],
            dict(os.environ),
            2,
        )


def test_cancellation_does_not_mask_command_teardown_failure(monkeypatch):
    started = threading.Event()

    def broken_runner(
        _command,
        _env,
        _timeout,
        *,
        cancellation_event=None,
    ):
        started.set()
        assert cancellation_event is not None
        assert cancellation_event.wait(timeout=5)
        raise RuntimeError("snapshot command teardown failed")

    monkeypatch.setattr(snapshot_refresh, "_run_command", broken_runner)

    async def scenario() -> None:
        command = asyncio.create_task(
            snapshot_refresh._await_command_completion(
                snapshot_refresh._run_command,
                [sys.executable, "-c", "pass"],
                os.environ,
                60,
            )
        )
        deadline = asyncio.get_running_loop().time() + 5
        while not started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("command teardown probe did not start")
            await asyncio.sleep(0.01)
        command.cancel()
        with pytest.raises(RuntimeError, match="command teardown failed"):
            await command

    asyncio.run(scenario())


def test_published_snapshot_uri_requires_one_content_addressed_target():
    expected = "s3://private-bucket/dev/" + "e" * 64
    assert snapshot_refresh._published_snapshot_uri(
        _publish_output(), base_uri="s3://private-bucket/dev"
    ) == expected

    for output in (
        "",
        "snapshot_uri=s3://other-bucket/dev/" + "e" * 64,
        "snapshot_uri=s3://private-bucket/dev/not-a-hash",
        _publish_output() + _publish_output(),
    ):
        try:
            snapshot_refresh._published_snapshot_uri(
                output, base_uri="s3://private-bucket/dev"
            )
        except RuntimeError:
            continue
        raise AssertionError("malformed publisher output was accepted")


def _configure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(snapshot_refresh.AUTO_REFRESH_ENABLED_ENV, "true")
    monkeypatch.setenv(snapshot_refresh.RECORD_ENABLED_ENV, "true")
    monkeypatch.setenv(snapshot_refresh.KMS_KEY_ID_ENV, "alias/dev-snapshot")
    monkeypatch.setenv(
        snapshot_refresh.PROVIDER_MODEL_IDS_ENV,
        '["provider/model-a","provider/model-b"]',
    )
    monkeypatch.setenv(snapshot_refresh.RUNTIME_SOURCE_ROOT_ENV, str(Path.cwd()))
    monkeypatch.setenv(
        snapshot_refresh.REFRESH_STATE_PATH_ENV,
        str(tmp_path / "state.json"),
    )
    monkeypatch.setenv(
        snapshot_refresh.REFRESH_WORK_ROOT_ENV,
        str(tmp_path / "work"),
    )
    monkeypatch.setenv(SNAPSHOT_URI_ENV, "s3://private-bucket/dev/current.json")


def test_cancellation_keeps_refresh_lock_and_workdir_until_command_is_gone(
    monkeypatch, tmp_path
):
    _configure(monkeypatch, tmp_path)
    runtime_root = tmp_path / "runtime"
    scripts_dir = runtime_root / "scripts"
    scripts_dir.mkdir(parents=True)
    recorder_pid_path = tmp_path / "recorder.pid"
    monkeypatch.setenv(snapshot_refresh.RUNTIME_SOURCE_ROOT_ENV, str(runtime_root))
    monkeypatch.setenv("SNAPSHOT_TEST_RECORDER_PID_PATH", str(recorder_pid_path))
    monkeypatch.setattr(snapshot_refresh, "COMMAND_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        snapshot_refresh,
        "COMMAND_TERMINATION_GRACE_SECONDS",
        0.75,
    )
    monkeypatch.setattr(snapshot_refresh, "COMMAND_KILL_WAIT_SECONDS", 3.0)
    (scripts_dir / "export_research_lab_dev_icp_inputs.py").write_text(
        "import argparse, json, pathlib\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--out-dir', required=True)\n"
        "parser.add_argument('--seed')\n"
        "parser.add_argument('--expected-private-model-manifest-hash')\n"
        "args = parser.parse_args()\n"
        "out = pathlib.Path(args.out_dir)\n"
        "out.mkdir(parents=True)\n"
        "document = {\n"
        "    'schema_version': 'research_lab.dev_icp_export.v2',\n"
        "    'items': [{'icp_ref': 'one'}],\n"
        "    'daily_bank_manifest': {'bank_size': 1},\n"
        "}\n"
        "(out / 'source_icps.json').write_text(json.dumps(document))\n",
        encoding="utf-8",
    )
    (scripts_dir / "record_research_lab_dev_snapshots.py").write_text(
        "import os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(os.environ['SNAPSHOT_TEST_RECORDER_PID_PATH']).write_text(\n"
        "    str(os.getpid())\n"
        ")\n"
        "os.close(1)\n"
        "os.close(2)\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    async def scenario() -> None:
        refresh = asyncio.create_task(
            snapshot_refresh.maybe_refresh_dev_snapshot(
                SimpleNamespace(),
                worker_index=0,
                tree_policy=TreePolicy(mode="active"),
                now=1000,
                readiness_loader=lambda _uri, **_kwargs: _ready(
                    ready=False,
                    reason="snapshot_not_ready",
                ),
                active_loader=active_loader,
            )
        )
        deadline = asyncio.get_running_loop().time() + 5
        while not recorder_pid_path.is_file():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("recorder process did not start")
            await asyncio.sleep(0.01)
        recorder_pid = int(recorder_pid_path.read_text(encoding="utf-8"))
        work_dirs = list((tmp_path / "work").glob("refresh-*"))
        assert len(work_dirs) == 1

        refresh.cancel()
        await asyncio.sleep(0.1)
        assert not refresh.done()
        assert work_dirs[0].is_dir()
        assert _pid_alive(recorder_pid)

        contender = await snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=1,
            tree_policy=TreePolicy(mode="active"),
            now=1001,
            readiness_loader=lambda _uri, **_kwargs: _ready(
                ready=False,
                reason="snapshot_not_ready",
            ),
            active_loader=active_loader,
        )
        assert contender == {"status": "skipped", "reason": "refresh_lock_held"}

        with pytest.raises(asyncio.CancelledError):
            await refresh
        assert not _pid_alive(recorder_pid)
        assert not any((tmp_path / "work").glob("refresh-*"))
        failure_state = json.loads(
            Path(os.environ[snapshot_refresh.REFRESH_STATE_PATH_ENV]).read_text(
                encoding="utf-8"
            )
        )
        assert failure_state["status"] == "failed"
        assert failure_state["last_error"] == (
            "CancelledError:snapshot refresh cancelled"
        )
        with snapshot_refresh._exclusive_refresh_lock(
            Path(os.environ[snapshot_refresh.REFRESH_STATE_PATH_ENV])
        ) as acquired:
            assert acquired is True

    asyncio.run(scenario())


def test_auto_refresh_defaults_on_and_explicit_false_disables(monkeypatch):
    monkeypatch.delenv(snapshot_refresh.AUTO_REFRESH_ENABLED_ENV, raising=False)
    assert snapshot_refresh.snapshot_auto_refresh_enabled() is True

    monkeypatch.setenv(snapshot_refresh.AUTO_REFRESH_ENABLED_ENV, "false")
    assert snapshot_refresh.snapshot_auto_refresh_enabled() is False

    monkeypatch.setenv(snapshot_refresh.AUTO_REFRESH_ENABLED_ENV, "invalid")
    assert snapshot_refresh.snapshot_auto_refresh_enabled() is False


def test_command_timeout_environment_is_finitely_capped(monkeypatch):
    assert snapshot_refresh.MAX_COMMAND_TIMEOUT_SECONDS == 2_622_300
    assert snapshot_refresh.COMMAND_TERMINATION_GRACE_SECONDS == (
        snapshot_refresh.SNAPSHOT_DOCKER_CLEANUP_TIMEOUT_SECONDS + 5.0
    )
    monkeypatch.setenv(
        snapshot_refresh.COMMAND_TIMEOUT_ENV,
        str(snapshot_refresh.MAX_COMMAND_TIMEOUT_SECONDS + 1),
    )
    assert snapshot_refresh._positive_env_int(
        snapshot_refresh.COMMAND_TIMEOUT_ENV,
        snapshot_refresh.DEFAULT_COMMAND_TIMEOUT_SECONDS,
        minimum=300,
        maximum=snapshot_refresh.MAX_COMMAND_TIMEOUT_SECONDS,
    ) == snapshot_refresh.MAX_COMMAND_TIMEOUT_SECONDS

    monkeypatch.setenv(snapshot_refresh.COMMAND_TIMEOUT_ENV, "inf")
    assert snapshot_refresh._positive_env_int(
        snapshot_refresh.COMMAND_TIMEOUT_ENV,
        snapshot_refresh.DEFAULT_COMMAND_TIMEOUT_SECONDS,
        minimum=300,
        maximum=snapshot_refresh.MAX_COMMAND_TIMEOUT_SECONDS,
    ) == snapshot_refresh.DEFAULT_COMMAND_TIMEOUT_SECONDS


def test_any_paid_worker_can_refresh_under_shared_lock(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    calls: list[list[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        calls.append(list(command))
        return _pipeline_output(command)

    readiness = iter(
        [
            _ready(ready=False, reason="snapshot_not_ready"),
            _ready(manifest_hash="sha256:" + "e" * 64),
            _ready(manifest_hash="sha256:" + "e" * 64),
        ]
    )
    refreshed = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=7,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: next(readiness),
            active_loader=active_loader,
        )
    )
    assert refreshed["status"] == "refreshed"
    assert len(calls) == 4

    disabled = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=8,
            tree_policy=TreePolicy(mode="off"),
            now=1000,
        )
    )
    assert disabled == {"status": "skipped", "reason": "tree_mode_not_active"}


def test_healthy_snapshot_check_is_persisted_across_restart(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    calls: list[Sequence[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command, _env, _timeout):
        calls.append(command)
        return ""

    first = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=1,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(),
            active_loader=active_loader,
        )
    )
    second = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=6,
            tree_policy=TreePolicy(mode="active"),
            now=1100,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(),
            active_loader=active_loader,
        )
    )
    assert first["status"] == "healthy"
    assert second == {"status": "skipped", "reason": "check_not_due"}
    assert not calls
    assert (tmp_path / "state.json").is_file()


def test_recent_model_a_check_does_not_delay_model_b_refresh(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    model_b = _active(
        image=IMAGE[:-1] + "e",
        commit="e" * 40,
        config_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
    )
    active = _active()
    commands: list[list[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return active

    first = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            readiness_loader=lambda _uri, **_kwargs: _ready(),
            active_loader=active_loader,
        )
    )
    assert first["status"] == "healthy"

    active = model_b
    readiness = iter(
        [
            _ready(),
            _ready(
                champion_image_digest=model_b.artifact.image_digest,
                source_commit=model_b.artifact.git_commit_sha,
                model_config_hash=model_b.artifact.config_hash,
                private_model_manifest_hash=model_b.artifact.manifest_hash,
                manifest_hash="sha256:" + "3" * 64,
            ),
            _ready(
                champion_image_digest=model_b.artifact.image_digest,
                source_commit=model_b.artifact.git_commit_sha,
                model_config_hash=model_b.artifact.config_hash,
                private_model_manifest_hash=model_b.artifact.manifest_hash,
                manifest_hash="sha256:" + "3" * 64,
            ),
        ]
    )

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        return _pipeline_output(command)

    second = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=6,
            tree_policy=TreePolicy(mode="active"),
            now=1100,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: next(readiness),
            active_loader=active_loader,
        )
    )

    assert second["status"] == "refreshed"
    assert len(commands) == 4


def test_failed_model_a_refresh_obeys_retry_cadence(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    calls: list[list[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        calls.append(list(command))
        raise RuntimeError("controlled recorder failure")

    first = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(
                ready=False,
                reason="not_ready",
            ),
            active_loader=active_loader,
        )
    )
    second = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1001,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(
                ready=False,
                reason="not_ready",
            ),
            active_loader=active_loader,
        )
    )

    assert first["status"] == "failed"
    assert first["active_manifest_hash"] == MODEL_MANIFEST_HASH
    assert second == {"status": "skipped", "reason": "check_not_due"}
    assert len(calls) == 1


def test_failed_model_a_refresh_does_not_delay_model_b(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    active = _active()
    model_b = _active(
        image=IMAGE[:-1] + "e",
        commit="e" * 40,
        config_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
    )
    calls: list[list[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return active

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        calls.append(list(command))
        raise RuntimeError("controlled recorder failure")

    first = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(
                ready=False,
                reason="not_ready",
            ),
            active_loader=active_loader,
        )
    )
    active = model_b
    second = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1001,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(
                ready=False,
                reason="active_model_mismatch",
            ),
            active_loader=active_loader,
        )
    )

    assert first["status"] == "failed"
    assert second["status"] == "failed"
    assert second["active_manifest_hash"] == model_b.artifact.manifest_hash
    assert len(calls) == 2


def test_failed_authority_read_obeys_short_retry_without_trusting_healthy_state(
    monkeypatch,
    tmp_path,
):
    _configure(monkeypatch, tmp_path)
    active_calls = 0

    async def failing_active_loader(*_args: Any, **_kwargs: Any):
        nonlocal active_calls
        active_calls += 1
        raise RuntimeError("active authority unavailable")

    state_path = tmp_path / "state.json"
    snapshot_refresh._write_state(
        state_path,
        {
            "schema_version": "research_lab.dev_snapshot_refresh_state.v1",
            "status": "failed",
            "last_check_unix": 1000,
            "last_error": "RuntimeError:active authority unavailable",
            "active_image_digest": IMAGE,
            "active_git_commit_sha": COMMIT,
            "active_config_hash": CONFIG_HASH,
            "active_manifest_hash": MODEL_MANIFEST_HASH,
        },
    )

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1001,
            active_loader=failing_active_loader,
        )
    )

    assert result == {
        "status": "skipped",
        "reason": "failed_check_retry_not_due",
    }
    assert active_calls == 1


def test_due_refresh_publishes_immutable_target_before_pointer(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    commands: list[list[str]] = []
    command_envs: list[dict[str, str]] = []
    command_timeouts: list[int] = []
    readiness = iter(
        [
            _ready(ready=False, reason="snapshot_not_ready"),
            _ready(manifest_hash="sha256:" + "e" * 64),
            _ready(manifest_hash="sha256:" + "e" * 64),
        ]
    )

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        command_envs.append(dict(_env))
        command_timeouts.append(_timeout)
        return _pipeline_output(command)

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: next(readiness),
            active_loader=active_loader,
        )
    )
    assert result["status"] == "refreshed"
    assert len(commands) == 4
    assert commands[0][1].endswith("export_research_lab_dev_icp_inputs.py")
    assert commands[1][1].endswith("record_research_lab_dev_snapshots.py")
    assert "--size" not in commands[1]
    timeout_index = commands[1].index("--timeout-seconds")
    assert commands[1][timeout_index + 1] == str(
        snapshot_refresh.DEFAULT_SNAPSHOT_ICP_TIMEOUT_SECONDS
    )
    expected_record_timeout = snapshot_refresh.snapshot_record_workflow_timeout_seconds(
        item_count=40,
        item_timeout_seconds=snapshot_refresh.DEFAULT_SNAPSHOT_ICP_TIMEOUT_SECONDS,
    )
    assert command_timeouts == [
        snapshot_refresh.DEFAULT_COMMAND_TIMEOUT_SECONDS,
        expected_record_timeout,
        snapshot_refresh.DEFAULT_COMMAND_TIMEOUT_SECONDS,
        snapshot_refresh.DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ]
    assert all(
        env[RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD["live_max_icps_per_node"]]
        == "5"
        for env in command_envs
    )
    assert commands[2][1].endswith("publish_research_lab_dev_snapshot.py")
    assert "--skip-current-pointer" in commands[2]
    assert "--skip-current-pointer" not in commands[3]
    assert not any((tmp_path / "work").glob("refresh-*"))


def test_missing_exported_bank_size_fails_before_paid_recording(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    commands: list[list[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(
        command: Sequence[str],
        _env: Mapping[str, str],
        _timeout: int,
    ) -> str:
        commands.append(list(command))
        return "export completed without a bank-size commitment\n"

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(
                ready=False,
                reason="snapshot_not_ready",
            ),
            active_loader=active_loader,
        )
    )

    assert result["status"] == "failed"
    assert "source_icps.json" in result["last_error"]
    assert len(commands) == 1
    assert commands[0][1].endswith("export_research_lab_dev_icp_inputs.py")


def test_exported_bank_over_safety_cap_fails_before_paid_recording(
    monkeypatch, tmp_path
):
    _configure(monkeypatch, tmp_path)
    commands: list[list[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(
        command: Sequence[str],
        _env: Mapping[str, str],
        _timeout: int,
    ) -> str:
        commands.append(list(command))
        return _pipeline_output(
            command,
            bank_size=snapshot_refresh.MAX_DEV_SNAPSHOT_BANK_ICP_COUNT + 1,
        )

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(
                ready=False,
                reason="snapshot_not_ready",
            ),
            active_loader=active_loader,
        )
    )

    assert result["status"] == "failed"
    assert result["last_error"] == (
        "RuntimeError:snapshot exporter bank size exceeds the safety cap"
    )
    assert len(commands) == 1
    assert commands[0][1].endswith("export_research_lab_dev_icp_inputs.py")


def test_due_refresh_uses_observed_model_provenance_without_manual_ids(
    monkeypatch, tmp_path
):
    _configure(monkeypatch, tmp_path)
    monkeypatch.delenv(snapshot_refresh.PROVIDER_MODEL_IDS_ENV)
    commands: list[list[str]] = []
    readiness = iter(
        [
            _ready(ready=False, reason="snapshot_not_ready"),
            _ready(manifest_hash="sha256:" + "e" * 64),
            _ready(manifest_hash="sha256:" + "e" * 64),
        ]
    )

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        return _pipeline_output(command)

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: next(readiness),
            active_loader=active_loader,
        )
    )

    assert result["status"] == "refreshed"
    record_command = commands[1]
    assert record_command[1].endswith("record_research_lab_dev_snapshots.py")
    assert "--provider-model-id" not in record_command


def test_active_model_change_keeps_existing_pointer_untouched(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    commands: list[list[str]] = []
    active_rows = iter(
        [_active(), _active(), _active(image=IMAGE[:-1] + "f")]
    )

    async def active_loader(*_args: Any, **_kwargs: Any):
        return next(active_rows)

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        return _pipeline_output(command)

    readiness = iter(
        [
            _ready(ready=False, reason="active_model_mismatch"),
            _ready(),
        ]
    )

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: next(readiness),
            active_loader=active_loader,
        )
    )
    assert result["status"] == "failed"
    assert "active private model changed" in result["last_error"]
    assert len(commands) == 2
    assert not any(
        command[1].endswith("publish_research_lab_dev_snapshot.py")
        and "--skip-current-pointer" not in command
        for command in commands
    )


def test_active_model_guard_cancels_stale_recording_before_publish(
    monkeypatch,
    tmp_path,
):
    _configure(monkeypatch, tmp_path)
    monkeypatch.setattr(
        snapshot_refresh,
        "ACTIVE_MODEL_GUARD_INTERVAL_SECONDS",
        0.01,
    )
    commands: list[list[str]] = []
    recorder_observed_cancel = threading.Event()
    active_calls = 0
    model_b = _active(
        image=IMAGE[:-1] + "e",
        commit="e" * 40,
        config_hash="sha256:" + "1" * 64,
        manifest_hash="sha256:" + "2" * 64,
    )

    async def active_loader(*_args: Any, **_kwargs: Any):
        nonlocal active_calls
        active_calls += 1
        return _active() if active_calls <= 2 else model_b

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        if command[1].endswith("export_research_lab_dev_icp_inputs.py"):
            return _pipeline_output(command)
        if not command[1].endswith("record_research_lab_dev_snapshots.py"):
            return "ok"
        cancel_file = Path(command[command.index("--cancel-file") + 1])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if cancel_file.is_file():
                recorder_observed_cancel.set()
                raise RuntimeError("recorder stopped at ICP boundary")
            time.sleep(0.005)
        raise AssertionError("active-model guard did not signal the recorder")

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(
                ready=False,
                reason="active_model_mismatch",
            ),
            active_loader=active_loader,
        )
    )

    assert result["status"] == "failed"
    assert "active private model changed during snapshot recording" in result["last_error"]
    assert recorder_observed_cancel.is_set()
    assert len(commands) == 2
    assert not any(
        command[1].endswith("publish_research_lab_dev_snapshot.py")
        for command in commands
    )
    assert not any((tmp_path / "work").glob("refresh-*"))


def test_active_model_guard_cancellation_does_not_mask_teardown_failure(
    monkeypatch,
    tmp_path,
):
    started = threading.Event()

    async def failed_teardown(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError as exc:
            raise RuntimeError("snapshot command teardown failed") from exc

    async def active_loader(*_args, **_kwargs):
        return _active()

    monkeypatch.setattr(
        snapshot_refresh,
        "_await_command_completion",
        failed_teardown,
    )

    async def scenario() -> None:
        guarded = asyncio.create_task(
            snapshot_refresh._run_record_command_with_active_guard(
                config=SimpleNamespace(),
                command_runner=lambda *_args, **_kwargs: "",
                command=[sys.executable, "-c", "pass"],
                env=dict(os.environ),
                timeout_seconds=60,
                active_loader=active_loader,
                expected_identity=snapshot_refresh._artifact_identity(_active()),
                cancel_file=tmp_path / "cancel-recording",
            )
        )
        deadline = asyncio.get_running_loop().time() + 5
        while not started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("guarded recorder did not start")
            await asyncio.sleep(0.01)
        guarded.cancel()
        with pytest.raises(RuntimeError, match="command teardown failed"):
            await guarded

    asyncio.run(scenario())


def test_utc_rollover_never_promotes_stale_snapshot_pointer(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    commands: list[list[str]] = []
    readiness = iter(
        [
            _ready(ready=False, reason="snapshot_not_ready"),
            _ready(
                ready=False,
                reason="snapshot_is_not_current_day_rebenchmark_bank",
            ),
        ]
    )

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        return _pipeline_output(command)

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: next(readiness),
            active_loader=active_loader,
        )
    )

    assert result["status"] == "failed"
    assert "snapshot_is_not_current_day_rebenchmark_bank" in result["last_error"]
    assert len(commands) == 3
    assert "--skip-current-pointer" in commands[-1]
    assert not any(
        command[1].endswith("publish_research_lab_dev_snapshot.py")
        and "--skip-current-pointer" not in command
        for command in commands
    )


def test_recording_failure_is_visible_and_never_promotes_pointer(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    commands: list[list[str]] = []

    async def active_loader(*_args: Any, **_kwargs: Any):
        return _active()

    def command_runner(command: Sequence[str], _env: Mapping[str, str], _timeout: int):
        commands.append(list(command))
        if command[1].endswith("record_research_lab_dev_snapshots.py"):
            raise RuntimeError("recording failed")
        return _pipeline_output(command)

    result = asyncio.run(
        snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            command_runner=command_runner,
            readiness_loader=lambda _uri, **_kwargs: _ready(ready=False, reason="not_ready"),
            active_loader=active_loader,
        )
    )
    assert result["status"] == "failed"
    assert "recording failed" in result["last_error"]
    assert len(commands) == 2
    assert not any(command[1].endswith("publish_research_lab_dev_snapshot.py") for command in commands)


def test_cross_process_lock_allows_only_one_simultaneous_check(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)

    async def _run():
        entered = asyncio.Event()
        release = asyncio.Event()

        calls = 0

        async def active_loader(*_args: Any, **_kwargs: Any):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                await release.wait()
            return _active()

        first = asyncio.create_task(
            snapshot_refresh.maybe_refresh_dev_snapshot(
                SimpleNamespace(),
                worker_index=0,
                tree_policy=TreePolicy(mode="active"),
                now=1000,
                readiness_loader=lambda _uri, **_kwargs: _ready(),
                active_loader=active_loader,
            )
        )
        await entered.wait()
        second = await snapshot_refresh.maybe_refresh_dev_snapshot(
            SimpleNamespace(),
            worker_index=0,
            tree_policy=TreePolicy(mode="active"),
            now=1000,
            readiness_loader=lambda _uri, **_kwargs: _ready(),
            active_loader=active_loader,
        )
        release.set()
        return await first, second

    first, second = asyncio.run(_run())
    assert first["status"] == "healthy"
    assert second == {"status": "skipped", "reason": "refresh_lock_held"}
