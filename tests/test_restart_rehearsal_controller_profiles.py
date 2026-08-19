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


def test_evidence_ownership_normalizer_is_exact_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller = _load_controller()
    commands: list[list[str]] = []
    monkeypatch.setattr(controller, "_run", lambda command: commands.append(command))
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir(mode=0o700)

    controller._normalize_evidence_ownership(
        "rehearsal-image",
        evidence_root=evidence_root,
        docker_platform="linux/arm64",
    )

    assert evidence_root.stat().st_mode & 0o777 == 0o700
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

    def fake_run(command):
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
