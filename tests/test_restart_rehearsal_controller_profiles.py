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


def test_prepush_workflow_uses_first_settled_component_slot() -> None:
    controller = _load_controller()
    validator_started = threading.Event()
    workflow_started = threading.Event()
    active: set[str] = set()
    peak_active = 0
    active_lock = threading.Lock()

    def enter(name: str) -> None:
        nonlocal peak_active
        with active_lock:
            active.add(name)
            peak_active = max(peak_active, len(active))

    def leave(name: str) -> None:
        with active_lock:
            active.remove(name)

    def gateway() -> None:
        enter("gateway")
        try:
            assert validator_started.wait(timeout=5)
        finally:
            leave("gateway")

    def validator() -> None:
        enter("validator")
        validator_started.set()
        try:
            assert workflow_started.wait(timeout=5)
        finally:
            leave("validator")

    def workflow() -> None:
        assert threading.current_thread() is threading.main_thread()
        enter("workflow")
        try:
            with active_lock:
                assert active == {"validator", "workflow"}
            workflow_started.set()
        finally:
            leave("workflow")

    stages: list[dict[str, object]] = []
    with controller._recording_fixture_stack(stages):
        workflow_results = controller._run_prepush_component_and_workflow_stages(
            component_actions=(
                ("gateway-forward-1", gateway),
                ("validator-forward-1", validator),
            ),
            workflow_action=("workflow-prepush", workflow),
            stages=stages,
        )
    stages.extend(workflow_results)

    assert peak_active == 2
    assert [item["stage"] for item in stages] == [
        "gateway-forward-1",
        "validator-forward-1",
        "fixture-cleanup",
        "workflow-prepush",
    ]
    assert all(item["status"] == "passed" for item in stages)


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


def test_instruction_files_define_fast_default_and_match() -> None:
    agents = (ROOT / "AGENTS.md").read_bytes()
    claude = (ROOT / "CLAUDE.md").read_bytes()

    assert agents == claude
    text = agents.decode("utf-8")
    assert "## Default verification: 5-10 minutes" in text
    assert "explicitly includes `un-accelerated` or" in text
    assert '"production-equivalent"' in text
    assert "--profile unaccelerated" in text
