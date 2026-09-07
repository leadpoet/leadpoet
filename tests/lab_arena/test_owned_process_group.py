"""Linux process-level tests for canonical Arena sidecar ownership."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from typing import Optional

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "manage_owned_process_group.py"
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc/self/stat").exists(),
    reason="process ownership uses Linux /proc identity",
)


PROCESS_SOURCE = """\
import signal
import time

running = True
def stop(*_args):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
print("ready", flush=True)
while running:
    time.sleep(0.05)
"""


def _make_entrypoint(tmp_path: Path, relative_path: str) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROCESS_SOURCE, encoding="utf-8")


def _start(
    tmp_path: Path, relative_path: str, arguments: list[str]
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-u", relative_path, *arguments],
        cwd=tmp_path,
        start_new_session=True,
        stdout=subprocess.PIPE,
    )


def _start_through_wrapper(
    tmp_path: Path, relative_path: str, arguments: list[str]
) -> subprocess.Popen[bytes]:
    wrapper = "import subprocess,sys; raise SystemExit(subprocess.call(sys.argv[1:]))"
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            wrapper,
            sys.executable,
            "-u",
            relative_path,
            *arguments,
        ],
        cwd=tmp_path,
        start_new_session=True,
        stdout=subprocess.PIPE,
    )


def _helper(
    action: str,
    tmp_path: Path,
    state_file: Path,
    relative_path: str,
    arguments: list[str],
    *,
    launch_pgid: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(HELPER),
        action,
        "--state-file",
        str(state_file),
        "--cwd",
        str(tmp_path),
        "--uid",
        str(os.getuid()),
    ]
    if launch_pgid is not None:
        command.extend(
            [
                "--launch-pgid",
                str(launch_pgid),
                "--discover-timeout-seconds",
                "2",
            ]
        )
    command.extend(["--", sys.executable, "-u", relative_path, *arguments])
    return subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=8
    )


def _await_ready(process: subprocess.Popen[bytes]) -> None:
    assert process.stdout is not None
    assert process.stdout.readline() == b"ready\n"


def _kill(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=3)


def _shell_function(script_name: str, function_name: str) -> str:
    script = (ROOT / script_name).read_text(encoding="utf-8")
    start = script.index(f"{function_name}() {{")
    end = script.index("\n}\n", start) + 3
    return script[start:end]


def _shell_process_helper_selector(script_name: str, role: str) -> str:
    script = (ROOT / script_name).read_text(encoding="utf-8")
    authority_variable = (
        "GATEWAY_RESTART_AUTHORITY_ROOT"
        if role == "gateway"
        else "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT"
    )
    state_variable = (
        "GATEWAY_CONTROLLER_PROCESS_STATE_FILE"
        if role == "gateway"
        else "VALIDATOR_CONTROLLER_PROCESS_STATE_FILE"
    )
    try:
        start = script.index(f'if [ -n "${authority_variable}" ]; then')
    except ValueError:
        legacy_assignment = "LAB_ARENA_PROCESS_HELPER="
        start = script.index(legacy_assignment)
        return script[start : script.index("\n", start) + 1]
    end_marker = f'\n{state_variable}='
    end = script.index("\n", script.index(end_marker, start) + 1) + 1
    return script[start:end]


@pytest.mark.parametrize("role", ["gateway", "validator"])
def test_current_controller_stops_historical_live_group_without_candidate_helper(
    tmp_path: Path, role: str
) -> None:
    historical_checkout = tmp_path / "historical-candidate"
    current_authority = tmp_path / "current-controller-authority"
    authority_helper = current_authority / "scripts" / HELPER.name
    authority_helper.parent.mkdir(parents=True)
    shutil.copy2(HELPER, authority_helper)
    state_file = tmp_path / "arena.json"

    if role == "gateway":
        script_name = "gw_restart.sh"
        function_name = "stop_lab_arena_service"
        relative_path = "scripts/run_lab_arena_service.py"
        environment_file = historical_checkout / "gateway.env"
        production_args = [
            "--environment-file",
            str(environment_file),
            "--host",
            "127.0.0.1",
            "--port",
            "8792",
        ]
        sidecar_args = [
            "--environment-file",
            str(environment_file),
            "--host",
            "127.0.0.1",
            "--port",
            "8793",
        ]
        variables = {
            "GATEWAY_PYTHON_BIN": sys.executable,
            "LEADPOET_REPO_ROOT": str(historical_checkout),
            "GATEWAY_RESTART_AUTHORITY_ROOT": str(current_authority),
            "GATEWAY_ENV_FILE": str(environment_file),
            "LAB_ARENA_SERVICE_STATE_FILE": str(state_file),
        }
    else:
        script_name = "validator_restart.sh"
        function_name = "stop_lab_arena_runner"
        relative_path = "scripts/run_lab_arena_runner.py"
        production_args = []
        sidecar_args = ["--round-id", "e2e-1"]
        variables = {
            "VALIDATOR_PYTHON_BIN": sys.executable,
            "VALIDATOR_ROOT": str(historical_checkout),
            "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT": str(current_authority),
            "LAB_ARENA_RUNNER_STATE_FILE": str(state_file),
        }

    shell = _shell_process_helper_selector(script_name, role)
    shell += _shell_function(script_name, function_name)
    shell += '\nsudo() { command "$@"; }\n'
    shell += f"\n{function_name}\n"
    _make_entrypoint(historical_checkout, relative_path)
    production = _start(historical_checkout, relative_path, production_args)
    unrelated_task_api = _start(
        historical_checkout, relative_path, sidecar_args
    )
    _await_ready(production)
    _await_ready(unrelated_task_api)
    assert not (historical_checkout / "scripts" / HELPER.name).exists()
    try:
        result = subprocess.run(
            ["bash", "-c", shell],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            env={
                **os.environ,
                **variables,
            },
        )

        assert result.returncode == 0, result.stderr
        production.wait(timeout=3)
        assert unrelated_task_api.poll() is None
        assert not state_file.exists()
    finally:
        _kill(production)
        _kill(unrelated_task_api)


@pytest.mark.parametrize("role", ["gateway", "validator"])
@pytest.mark.parametrize("invalid_helper", ["missing", "symlink"])
def test_controller_helper_is_rejected_before_shutdown(
    tmp_path: Path, role: str, invalid_helper: str
) -> None:
    historical_checkout = tmp_path / "historical-candidate"
    current_authority = tmp_path / "current-controller-authority"
    authority_helper = current_authority / "scripts" / HELPER.name
    authority_helper.parent.mkdir(parents=True)
    historical_checkout.mkdir()
    state_file = tmp_path / "arena.json"
    if invalid_helper == "symlink":
        authority_helper.symlink_to(HELPER)

    if role == "gateway":
        script_name = "gw_restart.sh"
        function_name = "stop_lab_arena_service"
        environment = {
            "GATEWAY_PYTHON_BIN": sys.executable,
            "LEADPOET_REPO_ROOT": str(historical_checkout),
            "GATEWAY_RESTART_AUTHORITY_ROOT": str(current_authority),
            "GATEWAY_ENV_FILE": str(historical_checkout / "gateway.env"),
            "LAB_ARENA_SERVICE_STATE_FILE": str(state_file),
        }
    else:
        script_name = "validator_restart.sh"
        function_name = "stop_lab_arena_runner"
        environment = {
            "VALIDATOR_PYTHON_BIN": sys.executable,
            "VALIDATOR_ROOT": str(historical_checkout),
            "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT": str(current_authority),
            "LAB_ARENA_RUNNER_STATE_FILE": str(state_file),
        }

    shell = "set -e\n" + _shell_process_helper_selector(script_name, role)
    shell += _shell_function(script_name, function_name)
    shell += '\nsudo() { command "$@"; }\n'
    shell += f'\n{function_name}\nprintf "shutdown-started\\n"\n'
    result = subprocess.run(
        ["bash", "-c", shell],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
        env={**os.environ, **environment},
    )

    assert result.returncode != 0
    assert "verified controller Lab Arena stop helper is unavailable" in result.stderr
    assert "shutdown-started" not in result.stdout


@pytest.mark.parametrize("role", ["gateway", "validator"])
def test_post_activation_start_records_with_current_controller_helper(
    tmp_path: Path, role: str
) -> None:
    historical_checkout = tmp_path / "historical-candidate"
    current_authority = tmp_path / "current-controller-authority"
    authority_helper = current_authority / "scripts" / HELPER.name
    authority_helper.parent.mkdir(parents=True)
    shutil.copy2(HELPER, authority_helper)
    state_file = tmp_path / "arena.json"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sudo = fake_bin / "sudo"
    fake_sudo.write_text("#!/bin/sh\nexec \"$@\"\n", encoding="utf-8")
    fake_sudo.chmod(0o755)

    if role == "gateway":
        script_name = "gw_restart.sh"
        function_name = "start_lab_arena_service"
        relative_path = "scripts/run_lab_arena_service.py"
        environment_file = historical_checkout / "gateway.env"
        production_args = [
            "--environment-file",
            str(environment_file),
            "--host",
            "127.0.0.1",
            "--port",
            "8792",
        ]
        environment = {
            "GATEWAY_PYTHON_BIN": sys.executable,
            "LEADPOET_REPO_ROOT": str(historical_checkout),
            "GATEWAY_RESTART_AUTHORITY_ROOT": str(current_authority),
            "GATEWAY_ENV_FILE": str(environment_file),
            "GATEWAY_LOG_ROOT": str(tmp_path / "logs"),
            "LAB_ARENA_SERVICE_LOG_FILE": str(tmp_path / "arena.log"),
            "LAB_ARENA_SERVICE_STATE_FILE": str(state_file),
            "LAB_ARENA_MODE": "shadow",
        }
        shell_suffix = '\ntimeout() { return 0; }\n'
    else:
        script_name = "validator_restart.sh"
        function_name = "start_lab_arena_runner"
        relative_path = "scripts/run_lab_arena_runner.py"
        production_args = []
        runsc = tmp_path / "runsc"
        runsc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        runsc.chmod(0o755)
        environment = {
            "VALIDATOR_PYTHON_BIN": sys.executable,
            "VALIDATOR_ROOT": str(historical_checkout),
            "VALIDATOR_ACTIVE_RELEASE_AUTHORITY_ROOT": str(current_authority),
            "LAB_ARENA_RUNNER_STATE_FILE": str(state_file),
            "LAB_ARENA_RUNNER_LOG_FILE": str(tmp_path / "arena.log"),
            "LAB_ARENA_MODE": "shadow",
            "LAB_ARENA_API_BASE_URL": "https://gateway.invalid",
            "LAB_ARENA_RUNSC_PATH": str(runsc),
        }
        shell_suffix = "\nsleep() { :; }\n"

    selector = _shell_process_helper_selector(script_name, role)
    start_function = _shell_function(script_name, function_name)
    _make_entrypoint(historical_checkout, relative_path)
    shell = "set -e\n" + selector + start_function + shell_suffix
    shell += f"\n{function_name}\n"
    state: dict[str, object] = {}
    try:
        result = subprocess.run(
            ["bash", "-c", shell],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            env={
                **os.environ,
                **environment,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
            },
        )

        assert result.returncode == 0, result.stderr
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["cwd"] == str(historical_checkout.resolve())
        if production_args:
            assert state["argv"][-len(production_args) :] == production_args
        else:
            assert state["argv"][-1] == relative_path
        assert not (historical_checkout / "scripts" / HELPER.name).exists()
    finally:
        if state_file.exists():
            _helper(
                "stop",
                historical_checkout,
                state_file,
                relative_path,
                production_args,
            )
        pgid = state.get("pgid")
        if isinstance(pgid, int):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("relative_path", "production_args", "sidecar_args"),
    [
        (
            "scripts/run_lab_arena_service.py",
            [
                "--environment-file",
                "gateway.env",
                "--host",
                "127.0.0.1",
                "--port",
                "8792",
            ],
            [
                "--environment-file",
                "gateway.env",
                "--host",
                "127.0.0.1",
                "--port",
                "8793",
            ],
        ),
        ("scripts/run_lab_arena_runner.py", [], ["--round-id", "e2e-1"]),
    ],
)
def test_n_minus_one_legacy_adoption_stops_only_exact_canonical_group(
    tmp_path: Path,
    relative_path: str,
    production_args: list[str],
    sidecar_args: list[str],
) -> None:
    _make_entrypoint(tmp_path, relative_path)
    production = _start(tmp_path, relative_path, production_args)
    sidecar = _start(tmp_path, relative_path, sidecar_args)
    _await_ready(production)
    _await_ready(sidecar)
    state_file = tmp_path / "state" / "arena.json"
    try:
        result = _helper("stop", tmp_path, state_file, relative_path, production_args)

        assert result.returncode == 0, result.stderr
        production.wait(timeout=3)
        assert sidecar.poll() is None
        assert not state_file.exists()
    finally:
        _kill(production)
        _kill(sidecar)


def test_recorded_process_group_stops_and_round_pinned_sidecar_survives(
    tmp_path: Path,
) -> None:
    relative_path = "scripts/run_lab_arena_runner.py"
    _make_entrypoint(tmp_path, relative_path)
    production = _start_through_wrapper(tmp_path, relative_path, [])
    sidecar = _start(tmp_path, relative_path, ["--round-id", "e2e-1"])
    _await_ready(production)
    _await_ready(sidecar)
    state_file = tmp_path / "arena.json"
    try:
        recorded = _helper(
            "record",
            tmp_path,
            state_file,
            relative_path,
            [],
            launch_pgid=production.pid,
        )
        assert recorded.returncode == 0, recorded.stderr
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["pgid"] == production.pid
        assert state["pid"] != production.pid
        assert not list(tmp_path.glob(f".{state_file.name}.*"))

        stopped = _helper("stop", tmp_path, state_file, relative_path, [])

        assert stopped.returncode == 0, stopped.stderr
        production.wait(timeout=3)
        assert sidecar.poll() is None
        assert not state_file.exists()
    finally:
        _kill(production)
        _kill(sidecar)


def test_reused_pid_identity_is_rejected_without_erasing_state(tmp_path: Path) -> None:
    relative_path = "scripts/run_lab_arena_runner.py"
    _make_entrypoint(tmp_path, relative_path)
    production = _start(tmp_path, relative_path, [])
    _await_ready(production)
    state_file = tmp_path / "arena.json"
    try:
        recorded = _helper(
            "record",
            tmp_path,
            state_file,
            relative_path,
            [],
            launch_pgid=production.pid,
        )
        assert recorded.returncode == 0, recorded.stderr
        state = json.loads(state_file.read_text(encoding="utf-8"))
        state["start_time_ticks"] += 1
        state_file.write_text(json.dumps(state), encoding="utf-8")

        stopped = _helper("stop", tmp_path, state_file, relative_path, [])

        assert stopped.returncode == 1
        assert "PID was reused" in stopped.stderr
        assert production.poll() is None
        assert state_file.exists()
    finally:
        _kill(production)


def test_owned_stale_state_is_removed_after_group_is_gone(tmp_path: Path) -> None:
    relative_path = "scripts/run_lab_arena_runner.py"
    _make_entrypoint(tmp_path, relative_path)
    production = _start(tmp_path, relative_path, [])
    _await_ready(production)
    state_file = tmp_path / "arena.json"
    recorded = _helper(
        "record",
        tmp_path,
        state_file,
        relative_path,
        [],
        launch_pgid=production.pid,
    )
    assert recorded.returncode == 0, recorded.stderr
    _kill(production)

    stopped = _helper("stop", tmp_path, state_file, relative_path, [])

    assert stopped.returncode == 0, stopped.stderr
    assert not state_file.exists()


def test_ambiguous_legacy_processes_fail_closed(tmp_path: Path) -> None:
    relative_path = "scripts/run_lab_arena_runner.py"
    _make_entrypoint(tmp_path, relative_path)
    first = _start(tmp_path, relative_path, [])
    second = _start(tmp_path, relative_path, [])
    _await_ready(first)
    _await_ready(second)
    state_file = tmp_path / "arena.json"
    try:
        stopped = _helper("stop", tmp_path, state_file, relative_path, [])

        assert stopped.returncode == 1
        assert "more than one exact legacy process" in stopped.stderr
        assert first.poll() is None
        assert second.poll() is None
        assert not state_file.exists()
    finally:
        _kill(first)
        _kill(second)


def test_unreadable_owned_process_keeps_state(tmp_path: Path, monkeypatch) -> None:
    from scripts import manage_owned_process_group

    state_file = tmp_path / "arena.json"
    state_file.write_text("owned\n", encoding="utf-8")

    def unreadable(_pid: int) -> None:
        raise manage_owned_process_group.ProcessUnreadable(
            "process identity is unreadable"
        )

    monkeypatch.setattr(manage_owned_process_group, "_read_process", unreadable)
    with pytest.raises(manage_owned_process_group.ProcessUnreadable):
        manage_owned_process_group._stop_state(
            state_file,
            {"pid": 123, "pgid": 123},
            term_seconds=0.0,
            kill_seconds=0.0,
        )

    assert state_file.read_text(encoding="utf-8") == "owned\n"
