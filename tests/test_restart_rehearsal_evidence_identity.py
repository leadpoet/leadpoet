from __future__ import annotations

import ast
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
from typing import Optional

import pytest

from Leadpoet.utils.subnet_epoch import read_subnet_epoch_snapshot
from tests.restart_rehearsal.verify_evidence import (
    _verify_production_identity,
)
from tests.restart_rehearsal import sitecustomize as rehearsal_boundary


SOURCE_PATH = Path("scripts/gateway_git_deploy.py")
ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, content: str, message: str) -> str:
    source = repo / SOURCE_PATH
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(content, encoding="utf-8")
    _git("add", SOURCE_PATH.as_posix(), cwd=repo)
    _git("commit", "-qm", message, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture
def transition_repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "rehearsal@example.invalid", cwd=repo)
    _git("config", "user.name", "Restart Rehearsal", cwd=repo)
    installed_content = "VERSION = 'installed'\n"
    installed_sha = _commit(repo, installed_content, "installed")
    candidate_sha = _commit(repo, "VERSION = 'candidate'\n", "candidate")
    return repo, installed_sha, candidate_sha, installed_content


def test_installed_source_identity_survives_expected_checkout_activation(
    transition_repo: tuple[Path, str, str, str],
) -> None:
    repo, installed_sha, candidate_sha, installed_content = transition_repo
    source = repo / SOURCE_PATH
    row = {
        "candidate_sha": candidate_sha,
        "source_commit": installed_sha,
        "source_git_path": SOURCE_PATH.as_posix(),
        "source_kind": "installed_checkout",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(
            installed_content.encode("utf-8")
        ).hexdigest(),
    }

    _verify_production_identity(
        row,
        installed_sha,
        candidate_sha,
        (repo,),
    )


def test_candidate_checkout_tampering_is_still_rejected(
    transition_repo: tuple[Path, str, str, str],
) -> None:
    repo, installed_sha, candidate_sha, _ = transition_repo
    source = repo / SOURCE_PATH
    candidate_blob = subprocess.run(
        ["git", "show", f"{candidate_sha}:{SOURCE_PATH.as_posix()}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    row = {
        "candidate_sha": candidate_sha,
        "source_commit": candidate_sha,
        "source_git_path": SOURCE_PATH.as_posix(),
        "source_kind": "candidate_checkout",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(candidate_blob).hexdigest(),
    }
    source.write_text("TAMPERED = True\n", encoding="utf-8")

    with pytest.raises(
        SystemExit,
        match="candidate production source bytes changed",
    ):
        _verify_production_identity(
            row,
            installed_sha,
            candidate_sha,
            (repo,),
        )


def test_gateway_enclave_uses_the_exact_transition_target_tree() -> None:
    launcher = (
        ROOT / "tests/restart_rehearsal/run_inside.sh"
    ).read_text(encoding="utf-8")
    service = (
        ROOT / "tests/restart_rehearsal/gateway_enclave_service.py"
    ).read_text(encoding="utf-8")

    assert (
        'git --git-dir=/srv/origin.git archive "$CANDIDATE_SHA" gateway'
        in launcher
    )
    assert (
        'REHEARSAL_GATEWAY_CANDIDATE_ROOT='
        '"$SELECTED_GATEWAY_SOURCE_ROOT/gateway"'
        in launcher
    )
    assert 'REHEARSAL_GATEWAY_CANDIDATE_ROOT="/source/gateway"' not in launcher
    assert 'controller_source = Path("/source").resolve()' in service
    assert '"/source/gateway"' not in service


def test_gateway_enclave_measured_runtime_adapter_is_strict() -> None:
    service = (
        ROOT / "tests/restart_rehearsal/gateway_enclave_service.py"
    ).read_text(encoding="utf-8")

    assert "production_prepare_cgroup" in service
    assert "_prepare_measured_cgroup_boundary" in service
    assert 'cgroup_layout="nitro_v1_controller_root"' in service
    assert "Nitro controller root unexpectedly exposes child limits" in service
    assert 'proc_lines.append(f"{hierarchy}:{controller}:/")' in service
    assert 'controller_root / "tasks"' in service
    assert 'current_pid = str(os.getpid())' in service
    assert 'delegated != "leadpoet-model"' in service
    assert '"--rootless=false"' in service
    assert '"--network=none"' in service
    assert '"--host-uds=open"' in service
    assert '"--platform=ptrace"' in service
    assert '"model_sandbox_self_test"' in service
    assert '"LEADPOET_MODEL_SOURCE_ROOT"' in service
    assert '"/leadpoet-model-sandboxes/lp-job-"' in service
    assert 'item.get("type") == "bind"' in service
    assert 'model sandbox measured rootfs inputs differ' in service
    assert 'raise ValueError("model sandbox runsc operation differs")' in service


def test_gateway_provider_adapter_tracks_production_transport_interface() -> None:
    production = ast.parse(
        (ROOT / "gateway/tee/provider_broker_v2.py").read_text(encoding="utf-8")
    )
    rehearsal = ast.parse(
        (ROOT / "tests/restart_rehearsal/sitecustomize.py").read_text(
            encoding="utf-8"
        )
    )

    production_call = next(
        item
        for node in production.body
        if isinstance(node, ast.ClassDef) and node.name == "HTTPXProviderTransport"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "__call__"
    )
    rehearsal_call = next(
        node
        for node in rehearsal.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_local_provider_transport"
    )

    production_keywords = {arg.arg for arg in production_call.args.kwonlyargs}
    rehearsal_keywords = {arg.arg for arg in rehearsal_call.args.kwonlyargs}
    assert production_keywords <= rehearsal_keywords


def test_local_chain_adapter_supports_exact_historical_epoch_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rehearsal_boundary, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_boundary,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    substrate = rehearsal_boundary._LocalSubstrate()
    target_epoch = rehearsal_boundary.CUTOVER_EPOCH_INDEX + 73
    boundary_block = rehearsal_boundary._subnet_epoch_transition_block(
        target_epoch
    )

    for block, expected_epoch in (
        (
            rehearsal_boundary.CUTOVER_BLOCK - 1,
            rehearsal_boundary.CUTOVER_EPOCH_INDEX - 1,
        ),
        (boundary_block - 1, target_epoch - 1),
        (boundary_block, target_epoch),
        (boundary_block + 1, target_epoch),
        (
            rehearsal_boundary.CURRENT_BLOCK,
            rehearsal_boundary.SUBNET_EPOCH_INDEX,
        ),
    ):
        block_hash = rehearsal_boundary._block_hash(block)
        assert substrate.get_block_number(block_hash) == block
        observed = {
            name: substrate.query(
                module="SubtensorModule",
                storage_function=name,
                params=[71],
                block_hash=block_hash,
            ).value
            for name in (
                "Tempo",
                "LastEpochBlock",
                "PendingEpochAt",
                "SubnetEpochIndex",
                "BlocksSinceLastStep",
            )
        }
        assert observed == rehearsal_boundary._subnet_epoch_state_at(block)
        assert observed["SubnetEpochIndex"] == expected_epoch
        assert observed["LastEpochBlock"] <= block
        assert observed["PendingEpochAt"] > block
        snapshot = read_subnet_epoch_snapshot(
            rehearsal_boundary._LocalSubtensor(network="finney"),
            netuid=71,
            block_number=block,
        )
        assert snapshot.current_block == block
        assert snapshot.subnet_epoch_index == expected_epoch
        assert snapshot.tempo > 0

    captured_current_hash = rehearsal_boundary._block_hash(
        rehearsal_boundary.CURRENT_BLOCK
    )
    rehearsal_boundary._BLOCK_NUMBERS_BY_HASH.clear()
    assert substrate.get_block_number(captured_current_hash) == (
        rehearsal_boundary.CURRENT_BLOCK
    )


@pytest.mark.parametrize(
    ("ambient_path", "expected_path_suffix"),
    (("/ambient/bin", "/ambient/bin"), (None, os.defpath)),
)
def test_n_minus_one_docker_reader_uses_an_isolated_bounded_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ambient_path: Optional[str],
    expected_path_suffix: str,
) -> None:
    from tests.restart_rehearsal import dynamic_docker_collision_workflow

    observed: dict[str, object] = {}

    class Child:
        def __init__(self) -> None:
            self.stdin = io.StringIO()

    child = Child()

    def popen(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return child

    event_path = tmp_path / "events.log"
    exact_root = tmp_path / "exact"
    launch_environment = {
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": "/ambient/operation.lock",
        "LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE": (
            "/ambient/admission.lock"
        ),
        "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "999",
        "LEADPOET_DOCKER_DAEMON_READY_TIMEOUT_SECONDS": "999",
    }
    if ambient_path is not None:
        launch_environment["PATH"] = ambient_path
    with monkeypatch.context() as scoped:
        if ambient_path is None:
            scoped.delenv("PATH", raising=False)
        scoped.setattr(
            dynamic_docker_collision_workflow.subprocess,
            "Popen",
            popen,
        )
        returned = (
            dynamic_docker_collision_workflow._run_exact_n_minus_one_source_reader(
                source_root=ROOT,
                exact_root=exact_root,
                environment=launch_environment,
                event_path=event_path,
                host_detect_path=tmp_path / "host-detect",
                image_digest="registry.invalid/model@sha256:" + "a" * 64,
                timeout_seconds=300,
                collision_timeout_seconds=7.1,
            )
        )

    environment = observed["environment"]
    assert isinstance(environment, dict)
    reader_lock = tmp_path / "n-minus-reader.lock"
    assert environment["LEADPOET_DOCKER_OPERATION_LOCK_FILE"] == str(reader_lock)
    assert environment["LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE"] == (
        f"{reader_lock}.admission"
    )
    assert environment["LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS"] == "8"
    assert environment["LEADPOET_DOCKER_DAEMON_READY_TIMEOUT_SECONDS"] == "8"
    reader_docker = tmp_path / "n-minus-reader-bin" / "docker"
    reader_ready_log = tmp_path / "n-minus-reader-docker-ready.log"
    assert environment["REHEARSAL_N_MINUS_READER_DOCKER_READY_LOG"] == str(
        reader_ready_log
    )
    assert environment["PATH"] == (
        f"{reader_docker.parent}{os.pathsep}{expected_path_suffix}"
    )
    assert all(environment["PATH"].split(os.pathsep))
    assert environment["PYTHONPATH"] == str(exact_root)
    assert returned is child
    assert (
        subprocess.run(
            [str(reader_docker), "info"],
            check=False,
            env=environment,
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            [str(reader_docker), "ps"],
            check=False,
            env=environment,
        ).returncode
        == 97
    )
    assert reader_ready_log.read_text(encoding="utf-8").splitlines() == ["info"]


def test_n_minus_one_docker_readiness_matches_exact_release_capability(
    tmp_path: Path,
) -> None:
    from tests.restart_rehearsal import dynamic_docker_collision_workflow

    exact_root = tmp_path / "exact"
    ready_log = tmp_path / "ready.log"
    assert dynamic_docker_collision_workflow._n_minus_one_docker_readiness_evidence(
        exact_root=exact_root,
        ready_log=ready_log,
    ) == (False, [])

    ready_log.write_text("info\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="readiness boundary differs"):
        dynamic_docker_collision_workflow._n_minus_one_docker_readiness_evidence(
            exact_root=exact_root,
            ready_log=ready_log,
        )
    ready_log.unlink()

    (exact_root / "research_lab").mkdir(parents=True)
    (exact_root / "research_lab/docker_operation_lock_v2.py").touch()
    with pytest.raises(RuntimeError, match="readiness boundary differs"):
        dynamic_docker_collision_workflow._n_minus_one_docker_readiness_evidence(
            exact_root=exact_root,
            ready_log=ready_log,
        )

    ready_log.write_text("info\n", encoding="utf-8")
    assert dynamic_docker_collision_workflow._n_minus_one_docker_readiness_evidence(
        exact_root=exact_root,
        ready_log=ready_log,
    ) == (True, ["info"])


def test_release_reuses_candidate_migrated_durable_boundary_state() -> None:
    controller = (
        ROOT / "scripts/run_local_restart_rehearsal.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "tests/restart_rehearsal/run_inside.sh"
    ).read_text(encoding="utf-8")

    assert 'dst=/rehearsal-durable-state"' in controller
    assert 'dst=/rehearsal-from-fixture-seed,readonly"' in controller
    assert "durable_state_root=durable_state_root" in controller
    assert re.search(
        r"from_fixture_seed_root=fixture_seeds\[\s*run_from\s*\]",
        controller,
    )
    assert re.search(
        r"durable_fixture_seed_root=fixture_seeds\[\s*candidate_sha\s*\]",
        controller,
    )
    assert 'REHEARSAL_DURABLE_SCHEMA_SHA:-' in launcher
    assert "REHEARSAL_DURABLE_SCHEMA_SHA is required" in launcher
    assert (
        '"$REHEARSAL_DURABLE_STATE_ROOT/postgrest-state.json"'
        in launcher
    )
    assert (
        '"$DURABLE_SCHEMA_SEED_ROOT/release-build-input.json"'
        in launcher
    )


def test_exact_launcher_evaluates_durable_schema_identity() -> None:
    launcher = ROOT / "tests/restart_rehearsal/run_inside.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "REHEARSAL_FROM_SHA": "a" * 40,
            "REHEARSAL_CANDIDATE_SHA": "b" * 40,
            "REHEARSAL_TRANSITION": "forward",
            "REHEARSAL_COMPONENT": "invalid",
            "REHEARSAL_DURABLE_SCHEMA_SHA": "b" * 40,
        },
    )

    assert result.returncode == 2
    assert (
        "REHEARSAL_COMPONENT must be gateway, validator, or workflow"
        in result.stderr
    )


def test_workflow_does_not_require_launcher_durable_schema_identity() -> None:
    launcher = ROOT / "tests/restart_rehearsal/run_inside.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "REHEARSAL_FROM_SHA": "a" * 40,
            "REHEARSAL_CANDIDATE_SHA": "b" * 40,
            "REHEARSAL_TRANSITION": "forward",
            "REHEARSAL_COMPONENT": "workflow",
            "REHEARSAL_PROFILE": "invalid",
        },
    )

    assert result.returncode == 2
    assert "REHEARSAL_PROFILE must be prepush or release" in result.stderr
    assert "REHEARSAL_DURABLE_SCHEMA_SHA is required" not in result.stderr


def test_exact_launcher_requires_durable_schema_identity() -> None:
    launcher = ROOT / "tests/restart_rehearsal/run_inside.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "REHEARSAL_FROM_SHA": "a" * 40,
            "REHEARSAL_CANDIDATE_SHA": "b" * 40,
            "REHEARSAL_TRANSITION": "forward",
            "REHEARSAL_COMPONENT": "gateway",
        },
    )

    assert result.returncode == 2
    assert "REHEARSAL_DURABLE_SCHEMA_SHA is required" in result.stderr
