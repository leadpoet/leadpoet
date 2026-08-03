from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess

import pytest

from tests.restart_rehearsal.verify_evidence import (
    _verify_production_identity,
)


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
    assert 'cgroup_layout="nitro_v1"' in service
    assert 'proc_lines.append(f"{hierarchy}:{controller}:/")' in service
    assert 'controller_root / "tasks"' in service
    assert 'current_pid = str(os.getpid())' in service
    assert 'delegated != "leadpoet-model"' in service
    assert '"--rootless=false"' in service
    assert '"--network=none"' in service
    assert '"--host-uds=open"' in service
    assert '"--platform=ptrace"' in service
    assert '"model_sandbox_self_test"' in service
    assert 'raise ValueError("model sandbox runsc operation differs")' in service


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
