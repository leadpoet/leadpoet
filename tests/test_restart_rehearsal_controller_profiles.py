from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import inspect
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = ROOT / "scripts" / "run_local_restart_rehearsal.py"


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "run_local_restart_rehearsal",
        CONTROLLER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_and_explicit_long_profiles_are_unambiguous() -> None:
    controller = _load_controller()

    assert controller.CLI_PROFILES == ("prepush", "unaccelerated")
    assert controller._runtime_profile("prepush") == "prepush"
    assert controller._runtime_profile("unaccelerated") == "release"
    with pytest.raises(ValueError, match="unsupported rehearsal profile"):
        controller._runtime_profile("release")


def test_old_release_cli_spelling_is_rejected_before_rehearsal() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CONTROLLER_PATH),
            "--profile",
            "release",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid choice: 'release'" in result.stderr
    assert "unaccelerated" in result.stderr


def _docker_info(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        ["docker", "info"],
        0,
        stdout=stdout,
        stderr="",
    )


def test_docker_resources_clamp_profile_to_available_capacity(monkeypatch) -> None:
    controller = _load_controller()
    calls = []

    def docker_info(command, *, capture=False):
        calls.append((list(command), capture))
        return _docker_info("2 2053640192\n")

    monkeypatch.setattr(
        controller,
        "_run",
        docker_info,
    )

    resources = controller._resolve_docker_resources("prepush")

    assert resources == {
        "available_cpus": "2",
        "available_memory_bytes": 2053640192,
        "effective_cpus": "2",
        "effective_memory_bytes": 2053640192,
        "requested_cpus": "4",
        "requested_memory": "7g",
        "requested_memory_bytes": 7 * 1024**3,
    }
    assert calls == [
        (
            [
                "docker",
                "info",
                "--format",
                "{{.NCPU}} {{.MemTotal}}",
            ],
            True,
        )
    ]


def test_docker_resources_leave_profile_below_capacity_unchanged(monkeypatch) -> None:
    controller = _load_controller()
    monkeypatch.setattr(
        controller,
        "_run",
        lambda *_args, **_kwargs: _docker_info("8 17179869184\n"),
    )

    resources = controller._resolve_docker_resources("prepush")

    assert resources["effective_cpus"] == "4"
    assert resources["effective_memory_bytes"] == 7 * 1024**3


@pytest.mark.parametrize(
    "capacity",
    (
        "not-a-number 2053640192\n",
        "0 2053640192\n",
        "nan 2053640192\n",
        "2 0\n",
        "2 invalid\n",
        "2\n",
    ),
)
def test_docker_resources_reject_invalid_capacity(monkeypatch, capacity) -> None:
    controller = _load_controller()
    monkeypatch.setattr(
        controller,
        "_run",
        lambda *_args, **_kwargs: _docker_info(capacity),
    )

    with pytest.raises(SystemExit, match="Docker"):
        controller._resolve_docker_resources("prepush")


def test_all_rehearsal_docker_runs_use_validated_resource_arguments() -> None:
    controller = _load_controller()

    for function in (
        controller._fixture_seed_command,
        controller._run_component,
        controller._run_workflow,
        controller._join_evidence,
        controller._run_python37_finalization_probe,
    ):
        assert "_docker_resource_args" in inspect.getsource(function)


def _write_fixture_seed(root: Path, target: str) -> Path:
    seed = root / "seed"
    seed.mkdir(exist_ok=True)
    (seed / "fixture-seed.json").write_text(
        json.dumps({"candidate_sha": target}),
        encoding="utf-8",
    )
    return seed


def _command_value(command, option: str) -> str:
    return command[command.index(option) + 1]


def _command_fixture_root(command) -> Path:
    mount = next(
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mount" and "dst=/rehearsal-state" in command[index + 1]
    )
    source = next(field for field in mount.split(",") if field.startswith("src="))
    return Path(source.removeprefix("src=")).parent


def test_fixture_seeds_overlap_with_two_cpus_and_keep_exact_mapping(
    monkeypatch,
    tmp_path,
) -> None:
    controller = _load_controller()
    targets = ("b" * 40, "a" * 40)
    observed: dict[str, tuple[str, str, str]] = {}
    roots: list[Path] = []

    class OverlapProcess:
        instances = []

        def __init__(self, command, *, cwd):
            self.command = list(command)
            self.cwd = cwd
            self.instances.append(self)
            target = _command_value(self.command, "--candidate-sha")
            root = _command_fixture_root(self.command)
            roots.append(root)
            observed[target] = (
                _command_value(self.command, "--cpus"),
                _command_value(self.command, "--memory"),
                next(
                    value
                    for value in self.command
                    if "dst=/opt/leadpoet/drand-cabi-v2" in value
                ),
            )

        def poll(self):
            assert len(self.instances) == 2
            return 0

    monkeypatch.setattr(controller.subprocess, "Popen", OverlapProcess)
    monkeypatch.setattr(controller, "_finalize_fixture_seed", _write_fixture_seed)
    stages = []
    with ExitStack() as stack:
        seeds = controller._prepare_fixture_seeds(
            "image",
            source_root=tmp_path,
            targets=targets,
            drand_artifacts={target: tmp_path / target for target in targets},
            docker_platform="linux/amd64",
            profile="prepush",
            docker_resources={
                "effective_cpus": "2",
                "effective_memory_bytes": 2053640192,
            },
            fixture_stack=stack,
            stages=stages,
        )
        assert list(seeds) == sorted(targets)
        assert all(path.is_dir() for path in seeds.values())
        assert all(cpu == "1" for cpu, _memory, _mount in observed.values())
        assert all(
            memory == "1026820096"
            for _cpu, memory, _mount in observed.values()
        )
        assert all(
            str(tmp_path / target) in observed[target][2]
            for target in targets
        )
        assert [item["stage"] for item in stages] == [
            f"fixture-seed-{target[:12]}" for target in sorted(targets)
        ]
    assert roots and all(not root.exists() for root in roots)


def test_fixture_seed_preparation_uses_one_worker_below_two_cpus(
    monkeypatch,
    tmp_path,
) -> None:
    controller = _load_controller()
    targets = ("a" * 40, "b" * 40)
    maximum_active = 0

    class SequentialProcess:
        active = 0

        def __init__(self, command, *, cwd):
            nonlocal maximum_active
            self.command = list(command)
            self.cwd = cwd
            self.finished = False
            self.__class__.active += 1
            maximum_active = max(maximum_active, self.__class__.active)

        def poll(self):
            if not self.finished:
                self.finished = True
                self.__class__.active -= 1
            return 0

    monkeypatch.setattr(controller.subprocess, "Popen", SequentialProcess)
    monkeypatch.setattr(controller, "_finalize_fixture_seed", _write_fixture_seed)
    with ExitStack() as stack:
        seeds = controller._prepare_fixture_seeds(
            "image",
            source_root=tmp_path,
            targets=targets,
            drand_artifacts={target: tmp_path for target in targets},
            docker_platform="linux/amd64",
            profile="prepush",
            docker_resources={
                "effective_cpus": "1.9",
                "effective_memory_bytes": 2053640192,
            },
            fixture_stack=stack,
            stages=[],
        )
    assert set(seeds) == set(targets)
    assert maximum_active == 1


def test_fixture_seed_failure_isolated_and_all_workspaces_cleaned(
    monkeypatch,
    tmp_path,
) -> None:
    controller = _load_controller()
    failed = "a" * 40
    succeeded = "b" * 40
    roots: list[Path] = []

    class IsolatedFailureProcess:
        def __init__(self, command, *, cwd):
            self.command = list(command)
            self.cwd = cwd
            self.target = _command_value(self.command, "--candidate-sha")
            roots.append(_command_fixture_root(self.command))

        def poll(self):
            return 17 if self.target == failed else 0

    monkeypatch.setattr(
        controller.subprocess,
        "Popen",
        IsolatedFailureProcess,
    )
    monkeypatch.setattr(controller, "_finalize_fixture_seed", _write_fixture_seed)
    stages = []
    with ExitStack() as stack:
        seeds = controller._prepare_fixture_seeds(
            "image",
            source_root=tmp_path,
            targets=(succeeded, failed),
            drand_artifacts={failed: tmp_path, succeeded: tmp_path},
            docker_platform="linux/amd64",
            profile="prepush",
            docker_resources={
                "effective_cpus": "2",
                "effective_memory_bytes": 2053640192,
            },
            fixture_stack=stack,
            stages=stages,
        )
        assert list(seeds) == [succeeded]
        assert seeds[succeeded].is_dir()
        assert [item["status"] for item in stages] == ["failed", "passed"]
    assert len(roots) == 2
    assert all(not root.exists() for root in roots)


def test_fixture_seed_missing_dependency_does_not_block_independent_target(
    monkeypatch,
    tmp_path,
) -> None:
    controller = _load_controller()
    missing = "a" * 40
    available = "b" * 40

    class ImmediateProcess:
        def __init__(self, command, *, cwd):
            self.command = list(command)
            self.cwd = cwd

        def poll(self):
            return 0

    monkeypatch.setattr(controller.subprocess, "Popen", ImmediateProcess)
    monkeypatch.setattr(controller, "_finalize_fixture_seed", _write_fixture_seed)
    stages = []
    with ExitStack() as stack:
        seeds = controller._prepare_fixture_seeds(
            "image",
            source_root=tmp_path,
            targets=(available, missing),
            drand_artifacts={available: tmp_path},
            docker_platform="linux/amd64",
            profile="prepush",
            docker_resources={
                "effective_cpus": "2",
                "effective_memory_bytes": 2053640192,
            },
            fixture_stack=stack,
            stages=stages,
        )

    assert list(seeds) == [available]
    assert stages[0] == {
        "blocked_by": [f"drand-artifact-{missing[:12]}"],
        "stage": f"fixture-seed-{missing[:12]}",
        "status": "unexercised",
    }
    assert stages[1]["stage"] == f"fixture-seed-{available[:12]}"
    assert stages[1]["status"] == "passed"


def test_fixture_seed_interruption_stops_exact_registry_processes(
    monkeypatch,
    tmp_path,
) -> None:
    controller = _load_controller()
    targets = ("a" * 40, "b" * 40)
    cleanup_commands = []
    roots: list[Path] = []
    finalizations = []

    class BlockingProcess:
        instances = []

        def __init__(self, command, *, cwd):
            self.command = list(command)
            self.cwd = cwd
            self.returncode = None
            self.stopped = threading.Event()
            self.terminated = False
            self.killed = False
            self.instances.append(self)
            roots.append(_command_fixture_root(self.command))

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15
            self.stopped.set()

        def kill(self):
            self.killed = True
            self.returncode = -9
            self.stopped.set()

        def wait(self, timeout=None):
            if not self.stopped.wait(timeout if timeout is not None else 5):
                raise subprocess.TimeoutExpired(self.command, timeout)
            return self.returncode

    def cleanup(command, **kwargs):
        cleanup_commands.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(controller.subprocess, "Popen", BlockingProcess)
    monkeypatch.setattr(controller.subprocess, "run", cleanup)
    monkeypatch.setattr(
        controller,
        "_finalize_fixture_seed",
        lambda *_args: finalizations.append(True),
    )

    started = time.monotonic()
    with ExitStack() as stack:
        with pytest.raises(
            controller.RehearsalTimeBudgetExceeded,
            match="1-second wall-clock budget",
        ):
            with controller._profile_time_limit(1):
                controller._prepare_fixture_seeds(
                    "image",
                    source_root=tmp_path,
                    targets=targets,
                    drand_artifacts={target: tmp_path for target in targets},
                    docker_platform="linux/amd64",
                    profile="prepush",
                    docker_resources={
                        "effective_cpus": "2",
                        "effective_memory_bytes": 2053640192,
                    },
                    fixture_stack=stack,
                    stages=[],
                )
    assert time.monotonic() - started < 3
    assert len(BlockingProcess.instances) == 2
    assert all(process.terminated for process in BlockingProcess.instances)
    assert finalizations == []
    assert cleanup_commands
    assert len(cleanup_commands) == 2
    assert all(
        command[:3] == ["docker", "rm", "--force"]
        and kwargs["timeout"] == 10.0
        for command, kwargs in cleanup_commands
    )
    assert all(not root.exists() for root in roots)


def test_fixture_seed_popen_interruption_removes_attempted_exact_name(
    monkeypatch,
    tmp_path,
) -> None:
    controller = _load_controller()
    target = "a" * 40
    received_commands = []
    cleanup_commands = []

    class InterruptedPopen:
        def __init__(self, command, *, cwd):
            received_commands.append((list(command), cwd))
            raise controller.RehearsalTimeBudgetExceeded(
                "injected Popen interruption"
            )

    def cleanup(command, **kwargs):
        cleanup_commands.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(controller.subprocess, "Popen", InterruptedPopen)
    monkeypatch.setattr(controller.subprocess, "run", cleanup)

    with ExitStack() as stack:
        with pytest.raises(
            controller.RehearsalTimeBudgetExceeded,
            match="injected Popen interruption",
        ):
            controller._prepare_fixture_seeds(
                "image",
                source_root=tmp_path,
                targets=(target,),
                drand_artifacts={target: tmp_path},
                docker_platform="linux/amd64",
                profile="prepush",
                docker_resources={
                    "effective_cpus": "2",
                    "effective_memory_bytes": 2053640192,
                },
                fixture_stack=stack,
                stages=[],
            )

    assert len(received_commands) == 1
    attempted_name = _command_value(received_commands[0][0], "--name")
    assert cleanup_commands == [
        (
            ["docker", "rm", "--force", attempted_name],
            {
                "capture_output": True,
                "check": False,
                "cwd": str(controller.REPO_ROOT),
                "text": True,
                "timeout": 10.0,
            },
        )
    ]


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="POSIX pthread signal masking is unavailable",
)
def test_fixture_seed_pending_alarm_runs_after_process_registration(
    monkeypatch,
    tmp_path,
) -> None:
    controller = _load_controller()
    target = "a" * 40
    cleanup_commands = []
    observed_masks = []

    class PendingAlarmProcess:
        instances = []

        def __init__(self, command, *, cwd):
            self.command = list(command)
            self.cwd = cwd
            self.returncode = None
            self.terminated = False
            self.instances.append(self)
            observed_masks.append(
                signal.pthread_sigmask(signal.SIG_BLOCK, set())
            )
            signal.raise_signal(signal.SIGALRM)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

    def cleanup(command, **kwargs):
        cleanup_commands.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(controller.subprocess, "Popen", PendingAlarmProcess)
    monkeypatch.setattr(controller.subprocess, "run", cleanup)

    with ExitStack() as stack:
        with pytest.raises(
            controller.RehearsalTimeBudgetExceeded,
            match="1-second wall-clock budget",
        ):
            with controller._profile_time_limit(1):
                controller._prepare_fixture_seeds(
                    "image",
                    source_root=tmp_path,
                    targets=(target,),
                    drand_artifacts={target: tmp_path},
                    docker_platform="linux/amd64",
                    profile="prepush",
                    docker_resources={
                        "effective_cpus": "2",
                        "effective_memory_bytes": 2053640192,
                    },
                    fixture_stack=stack,
                    stages=[],
                )

    assert len(PendingAlarmProcess.instances) == 1
    process = PendingAlarmProcess.instances[0]
    assert signal.SIGALRM in observed_masks[0]
    assert signal.SIGINT in observed_masks[0]
    assert process.terminated is True
    attempted_name = _command_value(process.command, "--name")
    assert [command for command, _kwargs in cleanup_commands] == [
        ["docker", "rm", "--force", attempted_name]
    ]


def test_fixture_seed_cleanup_failure_is_fail_closed(monkeypatch) -> None:
    controller = _load_controller()

    class StoppedProcess:
        def poll(self):
            return -15

    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr="daemon unavailable",
        ),
    )

    with pytest.raises(RuntimeError, match="container cleanup returned 1"):
        controller._stop_fixture_seed_processes(
            {"a" * 40: ("fixture-exact-name", StoppedProcess())}
        )


def test_stage_ledger_records_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _load_controller()
    observed = iter((10.0, 12.3456))
    monkeypatch.setattr(controller.time, "monotonic", lambda: next(observed))
    stages: list[dict[str, object]] = []

    passed, value = controller._run_independent_stage(
        stage="example",
        action=lambda: "ok",
        stages=stages,
    )

    assert passed is True
    assert value == "ok"
    assert stages == [
        {
            "duration_seconds": 2.346,
            "stage": "example",
            "status": "passed",
        }
    ]


def test_prepush_profile_has_a_hard_outer_deadline() -> None:
    controller = _load_controller()

    with pytest.raises(
        controller.RehearsalTimeBudgetExceeded,
        match="600-second wall-clock budget",
    ):
        with controller._profile_time_limit(600):
            signal.raise_signal(signal.SIGALRM)

    with controller._profile_time_limit(None):
        pass


def test_time_budget_is_not_swallowed_as_an_independent_stage_failure() -> None:
    controller = _load_controller()
    stages: list[dict[str, object]] = []

    with pytest.raises(controller.RehearsalTimeBudgetExceeded):
        with controller._profile_time_limit(600):
            controller._run_independent_stage(
                stage="blocking-launcher",
                action=lambda: signal.raise_signal(signal.SIGALRM),
                stages=stages,
            )

    assert stages == []


def test_workflow_preserves_distinct_n_minus_one_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    commands: list[list[str]] = []
    from_sha = "a" * 40
    candidate_sha = "b" * 40
    monkeypatch.setattr(controller, "_run", lambda command: commands.append(command))

    controller._run_workflow(
        "rehearsal-image",
        source_root=tmp_path / "source",
        evidence_root=tmp_path / "evidence",
        from_sha=from_sha,
        candidate_sha=candidate_sha,
        profile="prepush",
        docker_platform="linux/arm64",
        docker_resources={
            "available_cpus": "2",
            "available_memory_bytes": 2053640192,
            "effective_cpus": "2",
            "effective_memory_bytes": 2053640192,
        },
    )

    assert len(commands) == 1
    assert f"REHEARSAL_FROM_SHA={from_sha}" in commands[0]
    assert f"REHEARSAL_CANDIDATE_SHA={candidate_sha}" in commands[0]
    assert commands[0][commands[0].index("--cpus") + 1] == "2"
    assert commands[0][commands[0].index("--memory") + 1] == "2053640192"


def test_fixed_utility_resources_fail_closed_when_capacity_is_too_small() -> None:
    controller = _load_controller()

    with pytest.raises(SystemExit, match="fixed utility"):
        controller._docker_resource_args(
            {
                "available_cpus": "1",
                "available_memory_bytes": 500 * 1024**2,
            },
            fixed_cpus="1",
            fixed_memory="512m",
        )


def test_stage_summary_reports_requested_and_effective_resources(tmp_path) -> None:
    controller = _load_controller()
    resources = {
        "available_cpus": "2",
        "available_memory_bytes": 2053640192,
        "effective_cpus": "2",
        "effective_memory_bytes": 2053640192,
        "requested_cpus": "4",
        "requested_memory": "7g",
        "requested_memory_bytes": 7 * 1024**3,
    }

    output = controller._write_stage_summary(
        evidence_root=tmp_path,
        candidate_sha="b" * 40,
        elapsed_seconds=12.5,
        profile="prepush",
        stages=[],
        docker_resources=resources,
    )

    assert json.loads(output.read_text(encoding="utf-8"))[
        "docker_resources"
    ] == resources


def test_instruction_files_define_fast_default_and_match() -> None:
    agents = (ROOT / "AGENTS.md").read_bytes()
    claude = (ROOT / "CLAUDE.md").read_bytes()

    assert agents == claude
    text = agents.decode("utf-8")
    assert "## Default verification: 5-10 minutes" in text
    assert "explicitly includes `un-accelerated` or" in text
    assert '"production-equivalent"' in text
    assert "--profile unaccelerated" in text
