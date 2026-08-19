from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

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


def test_rehearsal_docker_config_is_private_helper_free_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_controller()
    monkeypatch.setenv("DOCKER_CONFIG", "/ambient/docker-config")

    with controller._isolated_docker_client_config() as root:
        config = root / "config.json"
        assert os.environ["DOCKER_CONFIG"] == str(root)
        assert root.stat().st_mode & 0o777 == 0o700
        assert config.stat().st_mode & 0o777 == 0o600
        assert config.read_text(encoding="utf-8") == '{"auths":{}}\n'
        assert "credsStore" not in config.read_text(encoding="utf-8")
        assert "credentialHelpers" not in config.read_text(encoding="utf-8")

    assert not root.exists()
    assert os.environ["DOCKER_CONFIG"] == "/ambient/docker-config"


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


def test_prepush_image_build_stays_main_thread_while_probe_overlaps() -> None:
    controller = _load_controller()
    image_started = threading.Event()
    probe_started = threading.Event()

    def image() -> str:
        assert threading.current_thread() is threading.main_thread()
        image_started.set()
        assert probe_started.wait(timeout=5)
        return "image"

    def probe() -> str:
        assert threading.current_thread() is not threading.main_thread()
        worker_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        assert signal.SIGALRM in worker_mask
        assert signal.SIGINT in worker_mask
        probe_started.set()
        assert image_started.wait(timeout=5)
        return "probe"

    assert controller._run_prepush_image_and_probe(
        image_action=image,
        probe_action=probe,
    ) == ("image", "probe")


class _BlockedProcess:
    pid = 12345

    def __init__(
        self,
        started: threading.Event,
        command,
        *,
        streaming: bool,
    ) -> None:
        self._started = started
        self.command = list(command)
        self.stdout = self if streaming else None
        self.returncode = None
        self.released = threading.Event()
        self.terminated = False
        self.killed = False
        self.reaped = False

    def __iter__(self):
        return self

    def __next__(self):
        self._started.set()
        assert self.released.wait(timeout=5)
        raise StopIteration

    def communicate(self):
        self._started.set()
        assert self.released.wait(timeout=5)
        return None, None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.released.set()

    def wait(self, timeout=None):
        if not self.released.wait(timeout=timeout):
            raise subprocess.TimeoutExpired(["blocked-worker"], timeout)
        self.reaped = True
        return self.returncode


def _install_blocked_popen(
    controller,
    monkeypatch: pytest.MonkeyPatch,
    started: threading.Event,
    *,
    streaming: bool,
) -> tuple[list[_BlockedProcess], list[str]]:
    processes: list[_BlockedProcess] = []
    removed_containers: list[str] = []

    def popen(command, **_kwargs):
        process = _BlockedProcess(started, command, streaming=streaming)
        processes.append(process)
        return process

    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    monkeypatch.setattr(controller, "_WORKER_TERMINATE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(
        controller,
        "_remove_worker_container",
        removed_containers.append,
    )
    return processes, removed_containers


def _install_routed_blocked_popen(
    controller,
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[str, threading.Event],
) -> tuple[list[_BlockedProcess], list[str]]:
    processes: list[_BlockedProcess] = []
    removed_containers: list[str] = []

    def popen(command, **kwargs):
        matches = [event for marker, event in routes.items() if marker in command]
        assert len(matches) == 1, command
        process = _BlockedProcess(
            matches[0],
            command,
            streaming=kwargs.get("bufsize") == 1,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    monkeypatch.setattr(controller, "_WORKER_TERMINATE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(
        controller,
        "_remove_worker_container",
        removed_containers.append,
    )
    return processes, removed_containers


def _run_test_component(controller, tmp_path: Path, component: str) -> None:
    controller._run_component(
        "image",
        source_root=tmp_path,
        component=component,
        from_sha="2" * 40,
        candidate_sha="3" * 40,
        transition="forward",
        evidence_root=tmp_path,
        drand_artifact_root=tmp_path,
        profile="prepush",
        docker_platform="linux/amd64",
        fixture_seed_root=tmp_path,
        from_fixture_seed_root=tmp_path,
        durable_fixture_seed_root=tmp_path,
        durable_state_root=tmp_path,
        durable_schema_sha="3" * 40,
        run_ordinal=1,
        gateway_worker_fleet_mode="active",
    )


def test_worker_container_cleanup_proves_continuous_full_bound_after_rm_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_controller()
    calls: list[list[str]] = []
    inspection_times: list[float] = []
    clock = [0.0]
    monkeypatch.setattr(controller, "_WORKER_DOCKER_CONVERGENCE_SECONDS", 0.2)
    monkeypatch.setattr(controller, "_WORKER_DOCKER_ABSENCE_OBSERVATIONS", 3)
    monkeypatch.setattr(controller, "_WORKER_DOCKER_POLL_SECONDS", 0.1)
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        controller.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    def run(command, **kwargs):
        calls.append(command)
        assert 0 < kwargs["timeout"] <= (
            controller._WORKER_DOCKER_CLEANUP_SECONDS
        )
        if command[2] == "rm":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        inspection_times.append(round(clock[0], 3))
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Error: No such container: exact-worker",
        )

    monkeypatch.setattr(controller.subprocess, "run", run)

    controller._remove_worker_container("exact-worker")

    assert calls == [
        ["docker", "container", "rm", "--force", "exact-worker"],
        ["docker", "container", "inspect", "exact-worker"],
        ["docker", "container", "rm", "--force", "exact-worker"],
        ["docker", "container", "inspect", "exact-worker"],
        ["docker", "container", "rm", "--force", "exact-worker"],
        ["docker", "container", "inspect", "exact-worker"],
    ]
    assert inspection_times == [0.0, 0.1, 0.2]


def test_worker_container_cleanup_fails_if_container_still_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_controller()
    monkeypatch.setattr(controller, "_WORKER_DOCKER_CONVERGENCE_SECONDS", 0.02)
    monkeypatch.setattr(controller, "_WORKER_DOCKER_POLL_SECONDS", 0.001)
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="container-id",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="absence did not converge"):
        controller._remove_worker_container("exact-worker")


def test_worker_container_cleanup_fails_on_t025_late_name_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_controller()
    clock = [0.0]
    inspection_times: list[float] = []
    monkeypatch.setattr(controller, "_WORKER_DOCKER_CONVERGENCE_SECONDS", 0.3)
    monkeypatch.setattr(controller, "_WORKER_DOCKER_ABSENCE_OBSERVATIONS", 3)
    monkeypatch.setattr(controller, "_WORKER_DOCKER_POLL_SECONDS", 0.05)
    monkeypatch.setattr(controller.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        controller.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    def run(command, **_kwargs):
        if command[2] == "rm":
            return subprocess.CompletedProcess(command, 0, "", "")
        observed_at = round(clock[0], 2)
        inspection_times.append(observed_at)
        if observed_at == 0.25:
            return subprocess.CompletedProcess(command, 0, "container-id", "")
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            "Error: No such container: exact-worker",
        )

    monkeypatch.setattr(controller.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="absence did not converge"):
        controller._remove_worker_container("exact-worker")

    assert 0.0 in inspection_times
    assert 0.1 in inspection_times
    assert 0.2 in inspection_times
    assert 0.25 in inspection_times
    assert inspection_times[-1] >= 0.3


def test_worker_run_communicate_failure_reaps_exact_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_controller()
    started = threading.Event()
    removed: list[str] = []
    processes: list[_BlockedProcess] = []

    class FailedProcess(_BlockedProcess):
        def communicate(self):
            raise RuntimeError("pipe failed")

    def popen(command, **_kwargs):
        process = FailedProcess(started, command, streaming=False)
        processes.append(process)
        return process

    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    monkeypatch.setattr(controller, "_WORKER_TERMINATE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(controller, "_remove_worker_container", removed.append)
    registry = controller._WorkerProcessRegistry()

    with controller._worker_process_scope(registry):
        with pytest.raises(RuntimeError, match="pipe failed"):
            controller._run(["docker", "run", "worker-image"])

    assert processes[0].terminated is True
    assert processes[0].killed is True
    assert processes[0].reaped is True
    name_index = processes[0].command.index("--name")
    assert removed == [processes[0].command[name_index + 1]]


def test_worker_registration_after_cancel_cleans_launch_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_controller()
    started = threading.Event()
    removed: list[str] = []
    monkeypatch.setattr(controller, "_WORKER_TERMINATE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(controller, "_remove_worker_container", removed.append)
    registry = controller._WorkerProcessRegistry()
    command, container_name = registry.prepare_command(
        ["docker", "run", "worker-image"]
    )
    process = _BlockedProcess(started, command, streaming=False)
    registry.cancel()

    with pytest.raises(controller.RehearsalTimeBudgetExceeded):
        registry.register(process, container_name)

    assert process.terminated is True
    assert process.killed is True
    assert process.reaped is True
    assert removed == [container_name]


@pytest.mark.parametrize(
    ("path", "signum"),
    (("run", signal.SIGALRM), ("component", signal.SIGINT)),
)
def test_main_spawn_registration_defers_signal_until_exact_cleanup(
    path: str,
    signum: signal.Signals,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    started = threading.Event()
    processes, removed = _install_blocked_popen(
        controller,
        monkeypatch,
        started,
        streaming=path == "component",
    )
    original_register = controller._WorkerProcessRegistry.register
    injected = False

    def signal_before_register(registry, process, container_name):
        nonlocal injected
        if not injected:
            injected = True
            signal.raise_signal(signum)
        return original_register(registry, process, container_name)

    monkeypatch.setattr(
        controller._WorkerProcessRegistry,
        "register",
        signal_before_register,
    )
    monkeypatch.setattr(
        controller,
        "_normalize_evidence_ownership",
        lambda *_args, **_kwargs: pytest.fail(
            "deadline must bypass ordinary normalization"
        ),
    )
    monkeypatch.setattr(
        controller,
        "_preserve_failure_evidence",
        lambda **_kwargs: pytest.fail(
            "deadline must bypass ordinary evidence preservation"
        ),
    )
    registry = controller._WorkerProcessRegistry()

    expected_exception = (
        controller.RehearsalTimeBudgetExceeded
        if signum == signal.SIGALRM
        else KeyboardInterrupt
    )
    with pytest.raises(expected_exception) as raised:
        with controller._profile_time_limit(600):
            with controller._worker_process_scope(registry):
                if path == "run":
                    controller._run(["docker", "run", "worker-image"])
                else:
                    _run_test_component(controller, tmp_path, "gateway")

    if signum == signal.SIGALRM:
        assert "600-second wall-clock budget" in str(raised.value)
    assert injected is True
    assert len(processes) == 1
    assert processes[0].terminated is True
    assert processes[0].killed is True
    assert processes[0].reaped is True
    name_index = processes[0].command.index("--name")
    assert removed == [processes[0].command[name_index + 1]]
    assert registry.cancel() == ()


def test_real_timer_waits_for_masked_worker_and_registered_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_controller()
    worker_started = threading.Event()
    worker_release = threading.Event()
    register_entered = threading.Event()
    register_completed = threading.Event()
    process_started = threading.Event()
    processes: list[_BlockedProcess] = []
    removed: list[str] = []
    worker_masks: list[set[signal.Signals]] = []

    def popen(command, **_kwargs):
        process = _BlockedProcess(process_started, command, streaming=False)
        processes.append(process)
        signal.setitimer(signal.ITIMER_REAL, 0.01)
        return process

    original_register = controller._WorkerProcessRegistry.register

    def delayed_register(registry, process, container_name):
        register_entered.set()
        controller.time.sleep(0.05)
        original_register(registry, process, container_name)
        register_completed.set()

    def probe() -> None:
        worker_masks.append(
            signal.pthread_sigmask(signal.SIG_BLOCK, set())
        )
        worker_started.set()
        assert worker_release.wait(timeout=1)

    def image() -> None:
        try:
            assert worker_started.wait(timeout=1)
            controller._run(["docker", "run", "image-build"])
        finally:
            worker_release.set()

    monkeypatch.setattr(controller.subprocess, "Popen", popen)
    monkeypatch.setattr(
        controller._WorkerProcessRegistry,
        "register",
        delayed_register,
    )
    monkeypatch.setattr(controller, "_WORKER_TERMINATE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(controller, "_remove_worker_container", removed.append)

    began = controller.time.monotonic()
    with pytest.raises(
        controller.RehearsalTimeBudgetExceeded,
        match="600-second wall-clock budget",
    ):
        with controller._profile_time_limit(600):
            controller._run_prepush_image_and_probe(
                image_action=image,
                probe_action=probe,
            )

    assert controller.time.monotonic() - began < 1
    assert register_entered.is_set()
    assert register_completed.is_set()
    assert worker_masks
    assert signal.SIGALRM in worker_masks[0]
    assert signal.SIGINT in worker_masks[0]
    assert len(processes) == 1
    assert processes[0].terminated is True
    assert processes[0].killed is True
    assert processes[0].reaped is True
    name_index = processes[0].command.index("--name")
    assert removed == [processes[0].command[name_index + 1]]


def test_workflow_deadline_bypasses_ordinary_failure_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    deadline = controller.RehearsalTimeBudgetExceeded("deadline")

    def fail_deadline(*_args, **_kwargs):
        raise deadline

    monkeypatch.setattr(controller, "_run", fail_deadline)
    monkeypatch.setattr(
        controller,
        "_normalize_evidence_ownership",
        lambda *_args, **_kwargs: pytest.fail(
            "deadline must bypass ordinary normalization"
        ),
    )
    monkeypatch.setattr(
        controller,
        "_preserve_failure_evidence",
        lambda **_kwargs: pytest.fail(
            "deadline must bypass ordinary evidence preservation"
        ),
    )

    with pytest.raises(controller.RehearsalTimeBudgetExceeded) as raised:
        controller._run_workflow(
            "image",
            source_root=tmp_path,
            evidence_root=tmp_path,
            from_sha="2" * 40,
            candidate_sha="3" * 40,
            profile="prepush",
            docker_platform="linux/amd64",
        )

    assert raised.value is deadline


def test_outer_evidence_cleanup_retains_path_without_replacing_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controller = _load_controller()
    evidence_root = tmp_path / "retained-evidence"
    evidence_root.mkdir()
    deadline = controller.RehearsalTimeBudgetExceeded("deadline")
    monkeypatch.setattr(
        controller.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(evidence_root),
    )
    monkeypatch.setattr(
        controller,
        "_normalize_evidence_ownership",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("normalization failed")
        ),
    )

    with pytest.raises(controller.RehearsalTimeBudgetExceeded) as raised:
        with controller._temporary_evidence_directory(
            "image",
            docker_platform="linux/amd64",
        ) as active_root:
            (active_root / "root-owned-artifact").write_text(
                "evidence\n",
                encoding="utf-8",
            )
            raise deadline

    assert raised.value is deadline
    assert evidence_root.is_dir()
    assert (evidence_root / "root-owned-artifact").is_file()
    captured = capsys.readouterr().err
    assert "REHEARSAL_EVIDENCE_RETAINED" in captured
    assert str(evidence_root) in captured


def test_prepush_main_alarm_cancels_and_reaps_blocked_probe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controller = _load_controller()
    started = threading.Event()
    finished = threading.Event()
    processes, removed_containers = _install_blocked_popen(
        controller,
        monkeypatch,
        started,
        streaming=False,
    )

    def fail_container_cleanup(container_name: str) -> None:
        removed_containers.append(container_name)
        raise RuntimeError("docker daemon unavailable")

    monkeypatch.setattr(
        controller,
        "_remove_worker_container",
        fail_container_cleanup,
    )

    def probe() -> None:
        try:
            controller._run(["docker", "run", "probe-image", "probe"])
        finally:
            finished.set()

    def alarm_image() -> None:
        assert started.wait(timeout=1)
        signal.raise_signal(signal.SIGALRM)

    began = controller.time.monotonic()
    with pytest.raises(
        controller.RehearsalTimeBudgetExceeded,
        match="600-second wall-clock budget",
    ):
        with controller._profile_time_limit(600):
            controller._run_prepush_image_and_probe(
                image_action=alarm_image,
                probe_action=probe,
            )

    assert controller.time.monotonic() - began < 1
    assert finished.is_set()
    assert len(processes) == 1
    assert processes[0].terminated is True
    assert processes[0].killed is True
    assert processes[0].reaped is True
    name_index = processes[0].command.index("--name")
    assert removed_containers == [processes[0].command[name_index + 1]]
    assert "REHEARSAL_WORKER_CLEANUP_FAILED" in capsys.readouterr().err


def test_prepush_workflow_alarm_cleans_both_named_runtime_containers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    validator_started = threading.Event()
    workflow_started = threading.Event()
    validator_finished = threading.Event()
    gateway_called = False
    processes, removed_containers = _install_routed_blocked_popen(
        controller,
        monkeypatch,
        {
            "REHEARSAL_COMPONENT=validator": validator_started,
            "workflow-image": workflow_started,
        },
    )
    monkeypatch.setattr(
        controller,
        "_normalize_evidence_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        controller,
        "_preserve_failure_evidence",
        lambda **_kwargs: tmp_path,
    )

    def gateway() -> None:
        nonlocal gateway_called
        gateway_called = True

    def validator() -> None:
        try:
            _run_test_component(controller, tmp_path, "validator")
        finally:
            validator_finished.set()

    def alarm_workflow() -> None:
        assert validator_started.wait(timeout=1)
        signal.setitimer(signal.ITIMER_REAL, 0.02)
        controller._run(["docker", "run", "workflow-image"])

    began = controller.time.monotonic()
    with pytest.raises(
        controller.RehearsalTimeBudgetExceeded,
        match="600-second wall-clock budget",
    ):
        with controller._profile_time_limit(600):
            controller._run_prepush_runtime_stages(
                preparation_action=lambda: (
                    ("gateway-forward-1", gateway),
                    ("validator-forward-1", validator),
                ),
                workflow_action=("workflow-prepush", alarm_workflow),
                expected_component_stages=(
                    "gateway-forward-1",
                    "validator-forward-1",
                ),
                stages=[],
            )

    assert controller.time.monotonic() - began < 1
    assert validator_finished.is_set()
    assert workflow_started.is_set()
    assert gateway_called is False
    assert len(processes) == 2
    assert all(process.terminated for process in processes)
    assert all(process.killed for process in processes)
    assert all(process.reaped for process in processes)
    names = {
        process.command[process.command.index("--name") + 1]
        for process in processes
    }
    assert len(names) == 2
    assert set(removed_containers) == names


def test_prepush_gateway_alarm_cleans_gateway_and_validator_without_next_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    gateway_started = threading.Event()
    validator_started = threading.Event()
    gateway_finished = threading.Event()
    validator_finished = threading.Event()
    workflow_calls: list[str] = []
    processes, removed_containers = _install_routed_blocked_popen(
        controller,
        monkeypatch,
        {
            "REHEARSAL_COMPONENT=gateway": gateway_started,
            "REHEARSAL_COMPONENT=validator": validator_started,
        },
    )
    monkeypatch.setattr(
        controller,
        "_normalize_evidence_ownership",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        controller,
        "_preserve_failure_evidence",
        lambda **_kwargs: tmp_path,
    )

    def gateway() -> None:
        try:
            assert validator_started.wait(timeout=1)
            signal.setitimer(signal.ITIMER_REAL, 0.02)
            _run_test_component(controller, tmp_path, "gateway")
        finally:
            gateway_finished.set()

    def validator() -> None:
        try:
            _run_test_component(controller, tmp_path, "validator")
        finally:
            validator_finished.set()

    began = controller.time.monotonic()
    with pytest.raises(
        controller.RehearsalTimeBudgetExceeded,
        match="600-second wall-clock budget",
    ):
        with controller._profile_time_limit(600):
            controller._run_prepush_runtime_stages(
                preparation_action=lambda: (
                    ("gateway-forward-1", gateway),
                    ("validator-forward-1", validator),
                ),
                workflow_action=(
                    "workflow-prepush",
                    lambda: workflow_calls.append("workflow"),
                ),
                expected_component_stages=(
                    "gateway-forward-1",
                    "validator-forward-1",
                ),
                stages=[],
            )

    assert controller.time.monotonic() - began < 1
    assert workflow_calls == ["workflow"]
    assert gateway_finished.is_set()
    assert validator_finished.is_set()
    assert len(processes) == 2
    assert all(process.terminated for process in processes)
    assert all(process.killed for process in processes)
    assert all(process.reaped for process in processes)
    names = {
        process.command[process.command.index("--name") + 1]
        for process in processes
    }
    assert len(names) == 2
    assert set(removed_containers) == names


def test_prepush_scheduler_runs_dependency_dag_with_two_slot_peak() -> None:
    controller = _load_controller()
    validator_started = threading.Event()
    workflow_started = threading.Event()
    gateway_started = threading.Event()
    active: set[str] = set()
    peak_active = 0
    events: list[str] = []
    active_lock = threading.Lock()

    def enter(name: str) -> None:
        nonlocal peak_active
        with active_lock:
            active.add(name)
            events.append(f"{name}-start")
            peak_active = max(peak_active, len(active))

    def leave(name: str) -> None:
        with active_lock:
            events.append(f"{name}-end")
            active.remove(name)

    def gateway() -> None:
        assert threading.current_thread() is threading.main_thread()
        enter("gateway")
        gateway_started.set()
        leave("gateway")

    def validator() -> None:
        assert threading.current_thread() is not threading.main_thread()
        enter("validator")
        validator_started.set()
        try:
            assert gateway_started.wait(timeout=5)
        finally:
            leave("validator")

    def prepare():
        enter("fixtures")
        try:
            assert workflow_started.wait(timeout=5)
        finally:
            leave("fixtures")
        return (
            ("gateway-forward-1", gateway),
            ("validator-forward-1", validator),
        )

    def workflow() -> None:
        assert threading.current_thread() is threading.main_thread()
        enter("workflow")
        workflow_started.set()
        try:
            assert validator_started.wait(timeout=5)
        finally:
            leave("workflow")

    stages: list[dict[str, object]] = []
    with controller._recording_fixture_stack(stages) as fixture_stack:
        fixture_stack.callback(lambda: events.append("fixture-close"))
        workflow_results = controller._run_prepush_runtime_stages(
            preparation_action=prepare,
            workflow_action=("workflow-prepush", workflow),
            expected_component_stages=(
                "gateway-forward-1",
                "validator-forward-1",
            ),
            stages=stages,
        )
    stages.extend(workflow_results)

    assert peak_active == 2
    assert events.index("workflow-start") < events.index("fixtures-end")
    assert events.index("fixtures-end") < events.index("validator-start")
    assert events.index("validator-start") < events.index("workflow-end")
    assert events.index("workflow-end") < events.index("gateway-start")
    assert events.index("gateway-start") < events.index("validator-end")
    assert events.index("gateway-end") < events.index("fixture-close")
    assert events.index("validator-end") < events.index("fixture-close")
    assert [item["stage"] for item in stages] == [
        "gateway-forward-1",
        "validator-forward-1",
        "fixture-cleanup",
        "workflow-prepush",
    ]
    assert all(item["status"] == "passed" for item in stages)


def test_prepush_scheduler_continues_after_workflow_and_validator_failures() -> None:
    controller = _load_controller()
    calls: list[str] = []

    def failed(name: str):
        def action() -> None:
            calls.append(name)
            raise RuntimeError(f"{name} failed")

        return action

    stages: list[dict[str, object]] = []
    workflow_results = controller._run_prepush_runtime_stages(
        preparation_action=lambda: (
            ("gateway-forward-1", lambda: calls.append("gateway")),
            ("validator-forward-1", failed("validator")),
        ),
        workflow_action=("workflow-prepush", failed("workflow")),
        expected_component_stages=(
            "gateway-forward-1",
            "validator-forward-1",
        ),
        stages=stages,
    )
    stages.extend(workflow_results)

    assert set(calls) == {"gateway", "validator", "workflow"}
    assert [item["status"] for item in stages] == [
        "passed",
        "failed",
        "failed",
    ]


def test_prepush_scheduler_records_preparation_failure_and_runs_workflow() -> None:
    controller = _load_controller()
    calls: list[str] = []
    stages: list[dict[str, object]] = []

    def fail_preparation():
        raise RuntimeError("fixture failed")

    workflow_results = controller._run_prepush_runtime_stages(
        preparation_action=fail_preparation,
        workflow_action=("workflow-prepush", lambda: calls.append("workflow")),
        expected_component_stages=(
            "gateway-forward-1",
            "validator-forward-1",
        ),
        stages=stages,
    )

    assert calls == ["workflow"]
    assert [item["stage"] for item in stages] == [
        "fixture-orchestration",
        "gateway-forward-1",
        "validator-forward-1",
    ]
    assert stages[0]["status"] == "failed"
    assert stages[1:] == [
        {
            "blocked_by": ["fixture-orchestration"],
            "stage": "gateway-forward-1",
            "status": "unexercised",
        },
        {
            "blocked_by": ["fixture-orchestration"],
            "stage": "validator-forward-1",
            "status": "unexercised",
        },
    ]
    assert workflow_results[0]["status"] == "passed"


def test_prepush_scheduler_does_not_swallow_preparation_budget_failure() -> None:
    controller = _load_controller()
    stages: list[dict[str, object]] = []

    def exceed_budget():
        raise controller.RehearsalTimeBudgetExceeded("budget exhausted")

    with pytest.raises(
        controller.RehearsalTimeBudgetExceeded,
        match="budget exhausted",
    ):
        controller._run_prepush_runtime_stages(
            preparation_action=exceed_budget,
            workflow_action=("workflow-prepush", lambda: None),
            expected_component_stages=(
                "gateway-forward-1",
                "validator-forward-1",
            ),
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
    )

    assert len(commands) == 1
    assert f"REHEARSAL_FROM_SHA={from_sha}" in commands[0]
    assert f"REHEARSAL_CANDIDATE_SHA={candidate_sha}" in commands[0]


def test_evidence_ownership_normalizer_is_exact_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    commands: list[list[str]] = []
    timeouts: list[float | None] = []

    def fake_run(command, *, timeout_seconds=None):
        commands.append(command)
        timeouts.append(timeout_seconds)

    monkeypatch.setattr(controller, "_run", fake_run)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir(mode=0o700)

    controller._normalize_evidence_ownership(
        "rehearsal-image",
        evidence_root=evidence_root,
        docker_platform="linux/arm64",
    )

    assert evidence_root.stat().st_mode & 0o777 == 0o700
    assert timeouts == [controller._EVIDENCE_NORMALIZATION_TIMEOUT_SECONDS]
    assert commands == [[
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/arm64",
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "128m",
        "--pids-limit",
        "32",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "CHOWN",
        "--cap-add",
        "DAC_READ_SEARCH",
        "--read-only",
        "--mount",
        f"type=bind,src={evidence_root},dst=/evidence",
        "--entrypoint",
        "/usr/bin/chown",
        "rehearsal-image",
        "--recursive",
        "--no-dereference",
        f"{os.getuid()}:{os.getgid()}",
        "/evidence",
    ]]


def test_fixture_seed_normalizes_before_host_inspection_and_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    candidate_sha = "c" * 40
    events: list[str] = []
    fixture_root: Path | None = None
    normalized = False

    def bind_source(command: list[str], destination: str) -> Path:
        suffix = f",dst={destination}"
        mount = next(
            argument
            for argument in command
            if argument.startswith("type=bind,src=")
            and argument.endswith(suffix)
        )
        return Path(mount.removeprefix("type=bind,src=").removesuffix(suffix))

    def fake_run(command: list[str]) -> None:
        nonlocal fixture_root
        events.append("generate")
        generated_state = bind_source(command, "/rehearsal-state")
        generated_config = bind_source(command, "/fixture-config")
        fixture_root = generated_state.parent
        (generated_state / "release-build-input.json").write_text(
            f'{{"commit_sha":"{candidate_sha}"}}\n',
            encoding="utf-8",
        )
        for name in (
            "validator-app",
            "gateway-enclave-build-identities",
            "gateway-attested-runtime",
        ):
            artifact = generated_state / name
            artifact.mkdir()
            (artifact / "artifact").write_text("fixture\n", encoding="utf-8")
        (generated_config / "config.json").write_text(
            "{}\n",
            encoding="utf-8",
        )

    def fake_normalize(
        tag: str,
        *,
        evidence_root: Path,
        docker_platform: str,
    ) -> None:
        nonlocal normalized
        assert tag == "rehearsal-image"
        assert evidence_root == fixture_root
        assert docker_platform == "linux/amd64"
        events.append("normalize")
        normalized = True

    def is_generated_path(path: Path) -> bool:
        return fixture_root is not None and (
            path == fixture_root or fixture_root in path.parents
        )

    original_is_file = Path.is_file
    original_is_dir = Path.is_dir
    original_read_text = Path.read_text
    original_copytree = controller.shutil.copytree
    original_copy2 = controller.shutil.copy2

    def guarded_is_file(path: Path) -> bool:
        if is_generated_path(path):
            assert normalized
        return original_is_file(path)

    def guarded_is_dir(path: Path) -> bool:
        if is_generated_path(path):
            assert normalized
        return original_is_dir(path)

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if is_generated_path(path):
            assert normalized
        return original_read_text(path, *args, **kwargs)

    def guarded_copytree(
        source: Path,
        destination: Path,
        *args: object,
        **kwargs: object,
    ):
        if is_generated_path(Path(source)):
            assert normalized
        return original_copytree(source, destination, *args, **kwargs)

    def guarded_copy2(
        source: Path,
        destination: Path,
        *args: object,
        **kwargs: object,
    ):
        if is_generated_path(Path(source)):
            assert normalized
        return original_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(controller, "_run", fake_run)
    monkeypatch.setattr(controller, "_normalize_evidence_ownership", fake_normalize)
    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    monkeypatch.setattr(Path, "is_dir", guarded_is_dir)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(controller.shutil, "copytree", guarded_copytree)
    monkeypatch.setattr(controller.shutil, "copy2", guarded_copy2)

    with controller._prepared_fixture_seed(
        "rehearsal-image",
        source_root=tmp_path / "source",
        candidate_sha=candidate_sha,
        drand_artifact_root=tmp_path / "drand",
        docker_platform="linux/amd64",
        profile="prepush",
    ) as seed:
        assert (seed / "fixture-seed.json").is_file()

    assert events == ["generate", "normalize"]


def test_fixture_seed_normalizes_after_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    events: list[str] = []

    def fail_generation(command: list[str]) -> None:
        assert command[:3] == ["docker", "run", "--rm"]
        events.append("generate-failed")
        raise subprocess.CalledProcessError(17, command)

    def fake_normalize(
        tag: str,
        *,
        evidence_root: Path,
        docker_platform: str,
    ) -> None:
        assert tag == "rehearsal-image"
        assert evidence_root.is_dir()
        assert docker_platform == "linux/amd64"
        events.append("normalize")

    monkeypatch.setattr(controller, "_run", fail_generation)
    monkeypatch.setattr(controller, "_normalize_evidence_ownership", fake_normalize)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        with controller._prepared_fixture_seed(
            "rehearsal-image",
            source_root=tmp_path / "source",
            candidate_sha="d" * 40,
            drand_artifact_root=tmp_path / "drand",
            docker_platform="linux/amd64",
            profile="prepush",
        ):
            pytest.fail("fixture generation failure must not yield a seed")

    assert raised.value.returncode == 17
    assert events == ["generate-failed", "normalize"]


def test_outer_evidence_normalizes_post_workflow_component_and_join_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _load_controller()
    events: list[str] = []

    def fake_normalize(
        tag: str,
        *,
        evidence_root: Path,
        docker_platform: str,
    ) -> None:
        assert tag == "rehearsal-image"
        assert docker_platform == "linux/arm64"
        assert (evidence_root / "workflow-complete").is_file()
        assert (
            evidence_root
            / "local-services"
            / "epochs"
            / "post-workflow-component"
        ).is_file()
        assert (evidence_root / "joined-evidence.json").is_file()
        events.append("outer-normalize")

    monkeypatch.setattr(
        controller,
        "_normalize_evidence_ownership",
        fake_normalize,
    )

    with controller._temporary_evidence_directory(
        "rehearsal-image",
        docker_platform="linux/arm64",
    ) as evidence_root:
        (evidence_root / "workflow-complete").write_text(
            "complete\n",
            encoding="utf-8",
        )
        events.append("workflow-complete")
        late_component_root = evidence_root / "local-services" / "epochs"
        late_component_root.mkdir(parents=True)
        (late_component_root / "post-workflow-component").write_text(
            "root-container-artifact\n",
            encoding="utf-8",
        )
        events.append("component-late-write")
        (evidence_root / "joined-evidence.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        events.append("join-late-write")

    assert events == [
        "workflow-complete",
        "component-late-write",
        "join-late-write",
        "outer-normalize",
    ]
    assert not evidence_root.exists()


def test_workflow_normalizes_evidence_before_preserving_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    commands: list[list[str]] = []
    events: list[str] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(1, command)

    def fake_preserve_failure_evidence(**_kwargs):
        assert len(commands) == 2
        events.append("preserved")

    monkeypatch.setattr(controller, "_run", fake_run)
    monkeypatch.setattr(
        controller,
        "_preserve_failure_evidence",
        fake_preserve_failure_evidence,
    )

    with pytest.raises(subprocess.CalledProcessError):
        controller._run_workflow(
            "rehearsal-image",
            source_root=tmp_path / "source",
            evidence_root=tmp_path / "evidence",
            from_sha="a" * 40,
            candidate_sha="b" * 40,
            profile="prepush",
            docker_platform="linux/arm64",
        )

    assert commands[1][-4:] == [
        "--recursive",
        "--no-dereference",
        f"{os.getuid()}:{os.getgid()}",
        "/evidence",
    ]
    assert events == ["preserved"]


def test_instruction_files_define_fast_default_and_match() -> None:
    agents = (ROOT / "AGENTS.md").read_bytes()
    claude = (ROOT / "CLAUDE.md").read_bytes()

    assert agents == claude
    text = agents.decode("utf-8")
    assert "## Default verification: 5-10 minutes" in text
    assert "explicitly includes `un-accelerated` or" in text
    assert '"production-equivalent"' in text
    assert "--profile unaccelerated" in text
