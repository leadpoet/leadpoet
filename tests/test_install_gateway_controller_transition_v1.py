from pathlib import Path
import fcntl
import os
import subprocess

import pytest

from scripts.install_gateway_controller_transition_v1 import (
    ControllerTransitionError,
    _require_inherited_lock,
    install_transition,
)


FILES = {
    "gw_restart.sh": b"#!/bin/bash\nexit 0\n",
    "scripts/gateway_git_deploy.py": b"deploy = True\n",
    "gateway/tee/host_memory_guard_v2.py": b"guard = True\n",
    "scripts/manage_owned_process_group.py": b"process = True\n",
}


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    for relative, payload in FILES.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o755 if relative == "gw_restart.sh" else 0o644)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-qm", "controller"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(["git", "remote", "add", "origin", str(repo)], cwd=repo, check=True)
    subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=repo, check=True)
    return repo, commit


def test_installs_exact_controller_atomically_and_is_repeatable(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    root = tmp_path / "controller"
    host = tmp_path / "gw_restart.sh"

    installed = install_transition(repo=repo, commit=commit, controller_root=root, host_restart=host)
    assert (root / "current").readlink() == Path(f"releases/{commit}")
    assert host.read_bytes() == FILES["gw_restart.sh"]
    assert oct(host.stat().st_mode & 0o777) == "0o700"
    assert install_transition(repo=repo, commit=commit, controller_root=root, host_restart=host) == installed


def test_rejects_dirty_or_non_main_candidate_and_preserves_controller(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    root = tmp_path / "controller"
    host = tmp_path / "gw_restart.sh"
    install_transition(repo=repo, commit=commit, controller_root=root, host_restart=host)
    before = host.read_bytes()
    (repo / "operator-note").write_text("keep\n")
    with pytest.raises(ControllerTransitionError, match="dirty"):
        install_transition(repo=repo, commit=commit, controller_root=root, host_restart=host)
    assert host.read_bytes() == before


def test_existing_release_collision_fails_closed(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    root = tmp_path / "controller"
    host = tmp_path / "gw_restart.sh"
    install_transition(repo=repo, commit=commit, controller_root=root, host_restart=host)
    (root / "releases" / commit / "gw_restart.sh").write_bytes(b"changed")
    with pytest.raises(ControllerTransitionError, match="differs"):
        install_transition(repo=repo, commit=commit, controller_root=root, host_restart=host)


def test_git_environment_override_is_rejected(tmp_path: Path, monkeypatch) -> None:
    repo, commit = _repo(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(repo / ".git"))
    with pytest.raises(ControllerTransitionError, match="overrides"):
        install_transition(
            repo=repo,
            commit=commit,
            controller_root=tmp_path / "controller",
            host_restart=tmp_path / "gw_restart.sh",
        )


def test_transition_wrapper_binds_origin_main_controller_and_migration() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "transition_gateway_controller_and_restart_v1.sh").read_text()
    assert "fetch --no-tags origin main" in source
    assert 'rev-parse origin/main' in source
    assert "install_gateway_controller_transition_v1.py" in source
    assert "203-retire-legacy-incentive-weight-bridge.sql" in source
    assert 'export GATEWAY_MIGRATION_203_SQL_SHA256="$migration_hash"' in source
    assert 'exec "$host_restart" --commit "$candidate"' in source


def test_inherited_descriptor_proves_live_exclusive_restart_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "restart.lock"
    with lock_path.open("a+b") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _require_inherited_lock(lock.fileno(), lock_path)
    wrong = tmp_path / "wrong.lock"
    wrong.write_bytes(b"")
    wrong.chmod(0o600)
    with wrong.open("a+b") as descriptor:
        with pytest.raises(ControllerTransitionError, match="invalid"):
            _require_inherited_lock(descriptor.fileno(), lock_path)
