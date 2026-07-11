from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts import gateway_git_deploy


@dataclass(frozen=True)
class GitFixture:
    remote: Path
    source: Path
    checkout: Path
    initial_sha: str


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, text: str, *, filename: str = "payload.txt") -> str:
    (repo / filename).write_text(text, encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", text)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def git_fixture(tmp_path: Path) -> GitFixture:
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(source)],
        check=True,
        capture_output=True,
    )
    _git(source, "config", "user.email", "gateway-tests@example.com")
    _git(source, "config", "user.name", "Gateway Tests")
    (source / "scripts").mkdir()
    (source / "gw_restart.sh").write_text(
        '#!/bin/bash\nGATEWAY_GIT_DEPLOY_PROTOCOL="1"\n',
        encoding="utf-8",
    )
    (source / "scripts" / "gateway_git_deploy.py").write_text(
        "# deployment helper fixture\n",
        encoding="utf-8",
    )
    _git(source, "add", "gw_restart.sh", "scripts/gateway_git_deploy.py")
    initial_sha = _commit(source, "initial")
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "-u", "origin", "main")
    subprocess.run(["git", "clone", str(remote), str(checkout)], check=True, capture_output=True)
    _git(checkout, "config", "user.email", "gateway-tests@example.com")
    _git(checkout, "config", "user.name", "Gateway Tests")
    return GitFixture(remote=remote, source=source, checkout=checkout, initial_sha=initial_sha)


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "plan.json",
        tmp_path / "deployments" / "gateway-current.json",
        tmp_path / "deployments" / "gateway-last-good.json",
    )


def _prepare(
    fixture: GitFixture,
    tmp_path: Path,
    *,
    branch: str = "main",
    deploy_commit: str = "",
    repo_url: str | None = None,
) -> dict:
    plan, manifest, last_good = _paths(tmp_path)
    return gateway_git_deploy.prepare_deployment(
        repo_root=fixture.checkout,
        repo_url=repo_url or str(fixture.remote),
        branch=branch,
        plan_file=plan,
        manifest_file=manifest,
        last_good_file=last_good,
        deploy_commit=deploy_commit,
    )


def test_noop_deployment_records_exact_commit_and_last_good(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    plan, manifest, last_good = _paths(tmp_path)
    prepared = _prepare(git_fixture, tmp_path)
    assert prepared["previous_sha"] == git_fixture.initial_sha
    assert prepared["target_sha"] == git_fixture.initial_sha
    assert prepared["remote_url"] == str(git_fixture.remote)
    assert _git(git_fixture.checkout, "rev-parse", "HEAD") == git_fixture.initial_sha

    activated = gateway_git_deploy.activate_deployment(plan_file=plan)
    assert activated["status"] == "activated"
    assert _git(git_fixture.checkout, "rev-parse", "HEAD") == git_fixture.initial_sha

    eif_root = tmp_path / "tee"
    eif_root.mkdir()
    pcr0 = "a" * 96
    (eif_root / "enclave-build-gateway_scoring_a.json").write_text(
        json.dumps({"Measurements": {"PCR0": pcr0}}),
        encoding="utf-8",
    )
    completed = gateway_git_deploy.finalize_deployment(
        plan_file=plan,
        status="succeeded",
        stage="health_verified",
        eif_root=eif_root,
    )
    assert completed["role_pcr0s"] == {"gateway_scoring_a": pcr0}
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "succeeded"
    assert json.loads(last_good.read_text(encoding="utf-8"))["target_sha"] == git_fixture.initial_sha


def test_fast_forward_is_fetched_before_checkout_activation(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    target = _commit(git_fixture.source, "second")
    _git(git_fixture.source, "push", "origin", "main")

    prepared = _prepare(git_fixture, tmp_path)
    assert prepared["target_sha"] == target
    assert _git(git_fixture.checkout, "rev-parse", "HEAD") == git_fixture.initial_sha

    gateway_git_deploy.activate_deployment(plan_file=_paths(tmp_path)[0])
    assert _git(git_fixture.checkout, "rev-parse", "HEAD") == target
    assert _git(git_fixture.checkout, "branch", "--show-current") == "main"


def test_configured_branch_is_selected(git_fixture: GitFixture, tmp_path: Path) -> None:
    _git(git_fixture.source, "checkout", "-b", "gateway-release")
    target = _commit(git_fixture.source, "release", filename="release.txt")
    _git(git_fixture.source, "push", "-u", "origin", "gateway-release")

    prepared = _prepare(git_fixture, tmp_path, branch="gateway-release")
    assert prepared["branch"] == "gateway-release"
    assert prepared["target_sha"] == target
    gateway_git_deploy.activate_deployment(plan_file=_paths(tmp_path)[0])
    assert _git(git_fixture.checkout, "branch", "--show-current") == "gateway-release"


def test_wrong_remote_fails_before_fetch(git_fixture: GitFixture, tmp_path: Path) -> None:
    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="origin"):
        _prepare(git_fixture, tmp_path, repo_url=str(tmp_path / "other.git"))
    assert _git(git_fixture.checkout, "rev-parse", "HEAD") == git_fixture.initial_sha


def test_fetch_failure_leaves_checkout_head_unchanged(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    missing_remote = tmp_path / "missing.git"
    _git(git_fixture.checkout, "remote", "set-url", "origin", str(missing_remote))
    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="fetch"):
        _prepare(git_fixture, tmp_path, repo_url=str(missing_remote))
    assert _git(git_fixture.checkout, "rev-parse", "HEAD") == git_fixture.initial_sha


def test_non_fast_forward_branch_is_rejected(git_fixture: GitFixture, tmp_path: Path) -> None:
    _git(git_fixture.source, "checkout", "--orphan", "rewritten")
    _git(git_fixture.source, "rm", "-rf", ".")
    _commit(git_fixture.source, "rewritten", filename="replacement.txt")
    _git(git_fixture.source, "push", "--force", "origin", "HEAD:main")

    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="fast-forward"):
        _prepare(git_fixture, tmp_path)
    assert _git(git_fixture.checkout, "rev-parse", "HEAD") == git_fixture.initial_sha


@pytest.mark.parametrize("state", ["unstaged", "staged", "untracked"])
def test_dirty_checkout_is_rejected(
    git_fixture: GitFixture,
    tmp_path: Path,
    state: str,
) -> None:
    if state == "unstaged":
        (git_fixture.checkout / "payload.txt").write_text("dirty", encoding="utf-8")
    else:
        path = git_fixture.checkout / "local.txt"
        path.write_text("dirty", encoding="utf-8")
        if state == "staged":
            _git(git_fixture.checkout, "add", "local.txt")
    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="dirty"):
        _prepare(git_fixture, tmp_path)


def test_reachable_full_sha_can_be_used_for_controlled_rollback(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    newer = _commit(git_fixture.source, "newer")
    _git(git_fixture.source, "push", "origin", "main")
    _prepare(git_fixture, tmp_path)
    gateway_git_deploy.activate_deployment(plan_file=_paths(tmp_path)[0])
    assert _git(git_fixture.checkout, "rev-parse", "HEAD") == newer

    rollback_dir = tmp_path / "rollback"
    prepared = _prepare(
        git_fixture,
        rollback_dir,
        deploy_commit=git_fixture.initial_sha,
    )
    assert prepared["mode"] == "pinned"
    gateway_git_deploy.activate_deployment(plan_file=_paths(rollback_dir)[0])
    assert _git(git_fixture.checkout, "rev-parse", "HEAD") == git_fixture.initial_sha
    assert _git(git_fixture.checkout, "branch", "--show-current") == ""


def test_rollback_pin_must_be_full_and_reachable(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="full 40-character"):
        _prepare(git_fixture, tmp_path / "short", deploy_commit=git_fixture.initial_sha[:12])

    _git(git_fixture.source, "checkout", "--orphan", "unrelated")
    _git(git_fixture.source, "rm", "-rf", ".")
    unrelated = _commit(git_fixture.source, "unrelated", filename="unrelated.txt")
    _git(git_fixture.source, "push", "-u", "origin", "unrelated")
    _git(git_fixture.checkout, "fetch", "origin", "unrelated")
    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="not reachable"):
        _prepare(git_fixture, tmp_path / "unreachable", deploy_commit=unrelated)


def test_target_must_support_restart_handoff_protocol(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    (git_fixture.source / "gw_restart.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    _git(git_fixture.source, "add", "gw_restart.sh")
    _git(git_fixture.source, "commit", "-m", "remove protocol")
    _git(git_fixture.source, "push", "origin", "main")
    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="handoff protocol"):
        _prepare(git_fixture, tmp_path)


def test_activation_rejects_checkout_changed_after_prepare(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    second = _commit(git_fixture.source, "second")
    _git(git_fixture.source, "push", "origin", "main")
    _prepare(git_fixture, tmp_path)
    _git(git_fixture.checkout, "checkout", "--detach", second)

    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="HEAD changed"):
        gateway_git_deploy.activate_deployment(plan_file=_paths(tmp_path)[0])


def test_activation_rejects_remote_tracking_ref_changed_after_prepare(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    _commit(git_fixture.source, "second")
    _git(git_fixture.source, "push", "origin", "main")
    _prepare(git_fixture, tmp_path)

    _commit(git_fixture.source, "third")
    _git(git_fixture.source, "push", "origin", "main")
    _git(git_fixture.checkout, "fetch", "origin", "main")
    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="changed"):
        gateway_git_deploy.activate_deployment(plan_file=_paths(tmp_path)[0])


def test_prepare_cli_reads_repo_and_branch_from_hydrated_env_file(
    git_fixture: GitFixture,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _git(git_fixture.source, "checkout", "-b", "gateway-release")
    target = _commit(git_fixture.source, "release", filename="release.txt")
    _git(git_fixture.source, "push", "-u", "origin", "gateway-release")
    env_file = tmp_path / "gateway.env"
    env_file.write_text(
        f"GITHUB_REPO_URL={git_fixture.remote}\nGITHUB_BRANCH=gateway-release\n",
        encoding="utf-8",
    )
    for key in ("GITHUB_REPO_URL", "GITHUB_BRANCH", "GATEWAY_DEPLOY_COMMIT"):
        monkeypatch.delenv(key, raising=False)
    plan, manifest, last_good = _paths(tmp_path)
    assert (
        gateway_git_deploy.main(
            [
                "prepare",
                "--repo-root",
                str(git_fixture.checkout),
                "--env-file",
                str(env_file),
                "--plan-file",
                str(plan),
                "--manifest-file",
                str(manifest),
                "--last-good-file",
                str(last_good),
            ]
        )
        == 0
    )
    assert json.loads(plan.read_text(encoding="utf-8"))["target_sha"] == target


def test_failed_finalize_does_not_replace_last_good(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    plan, _, last_good = _paths(tmp_path)
    _prepare(git_fixture, tmp_path)
    gateway_git_deploy.activate_deployment(plan_file=plan)
    failed = gateway_git_deploy.finalize_deployment(
        plan_file=plan,
        status="failed",
        stage="worker_import_preflight",
        eif_root=tmp_path / "missing-tee",
    )
    assert not last_good.exists()
    assert failed["role_pcr0s"] == {}
    assert json.loads(plan.read_text(encoding="utf-8"))["stage"] == "worker_import_preflight"


def test_remote_credentials_are_not_recorded() -> None:
    assert (
        gateway_git_deploy._sanitize_remote(
            "https://token@example.com/leadpoet/leadpoet.git?access_token=secret"
        )
        == "https://example.com/leadpoet/leadpoet.git"
    )
    assert (
        gateway_git_deploy._sanitize_remote("git@example.com:leadpoet/leadpoet.git")
        == "example.com:leadpoet/leadpoet.git"
    )
