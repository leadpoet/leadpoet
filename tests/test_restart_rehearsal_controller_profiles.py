from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import signal
import subprocess
import sys

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
        controller._prepared_fixture_seed,
        controller._run_component,
        controller._run_workflow,
        controller._join_evidence,
        controller._run_python37_finalization_probe,
    ):
        assert "_docker_resource_args" in inspect.getsource(function)


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
