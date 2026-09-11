from __future__ import annotations

import json
import subprocess
import tarfile
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
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
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


def test_fetch_branch_retries_transient_transport_failure(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []
    sleeps: list[int] = []

    def fake_run_git(_repo_root: Path, *args: str, **_kwargs) -> str:
        calls.append(args)
        if len(calls) == 1:
            raise gateway_git_deploy.GatewayGitDeployError(
                "git fetch failed: RPC failed; HTTP 503 curl 22; "
                "fatal: expected 'acknowledgments'"
            )
        return ""

    monkeypatch.setattr(gateway_git_deploy, "_run_git", fake_run_git)
    monkeypatch.setattr(gateway_git_deploy.time, "sleep", sleeps.append)

    gateway_git_deploy._fetch_branch_with_retry(Path("/repo"), "main")

    assert calls == [
        ("fetch", "--prune", "origin", "+refs/heads/main:refs/remotes/origin/main"),
        ("fetch", "--prune", "origin", "+refs/heads/main:refs/remotes/origin/main"),
    ]
    assert sleeps == [1]


def test_fetch_branch_does_not_retry_permanent_failure(monkeypatch) -> None:
    calls = 0

    def fake_run_git(_repo_root: Path, *_args: str, **_kwargs) -> str:
        nonlocal calls
        calls += 1
        raise gateway_git_deploy.GatewayGitDeployError(
            "git fetch failed: remote: Repository not found"
        )

    monkeypatch.setattr(gateway_git_deploy, "_run_git", fake_run_git)
    monkeypatch.setattr(
        gateway_git_deploy.time,
        "sleep",
        lambda _seconds: pytest.fail("permanent fetch failure was retried"),
    )

    with pytest.raises(
        gateway_git_deploy.GatewayGitDeployError,
        match="Repository not found",
    ):
        gateway_git_deploy._fetch_branch_with_retry(Path("/repo"), "main")

    assert calls == 1


def test_fetch_branch_exhausts_bounded_transient_retries(monkeypatch) -> None:
    calls = 0
    sleeps: list[int] = []

    def fake_run_git(_repo_root: Path, *_args: str, **_kwargs) -> str:
        nonlocal calls
        calls += 1
        raise gateway_git_deploy.GatewayGitDeployError(
            "git fetch failed: HTTP 503"
        )

    monkeypatch.setattr(gateway_git_deploy, "_run_git", fake_run_git)
    monkeypatch.setattr(gateway_git_deploy.time, "sleep", sleeps.append)

    with pytest.raises(gateway_git_deploy.GatewayGitDeployError, match="HTTP 503"):
        gateway_git_deploy._fetch_branch_with_retry(Path("/repo"), "main")

    assert calls == gateway_git_deploy.GIT_FETCH_MAX_ATTEMPTS
    assert sleeps == [1, 2, 4]


def _materialize_commit(
    repo: Path,
    commit: str,
    destination: Path,
) -> None:
    archive = destination.parent / f"{destination.name}.tar"
    destination.mkdir()
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            "--output",
            str(archive),
            commit,
        ],
        check=True,
        capture_output=True,
    )
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination)


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
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        gateway_git_deploy,
        "_installed_release_role_pcr0s",
        lambda _root, _target: {"gateway_scoring": pcr0},
    )
    (eif_root / "enclave-build-gateway_scoring.json").write_text(
        json.dumps({"Measurements": {"PCR0": pcr0}}),
        encoding="utf-8",
    )
    (eif_root / "enclave-build-gateway_autoresearch.json").write_text(
        json.dumps({"Measurements": {"PCR0": "b" * 96}}),
        encoding="utf-8",
    )
    completed = gateway_git_deploy.finalize_deployment(
        plan_file=plan,
        status="succeeded",
        stage="health_verified",
        eif_root=eif_root,
    )
    assert completed["role_pcr0s"] == {"gateway_scoring": pcr0}
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "succeeded"
    assert json.loads(last_good.read_text(encoding="utf-8"))["target_sha"] == git_fixture.initial_sha


def _use_fixture_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.tee import verify_release_artifacts_v2 as verifier

    def read_fixture_measurement(path: Path) -> str:
        measurement = path.with_name(
            path.name.replace("tee-enclave-", "enclave-build-").replace(
                ".eif", ".json"
            )
        )
        return verifier._pcr0_from_build_output(measurement)

    monkeypatch.setattr(verifier, "_pcr0_from_eif", read_fixture_measurement)


def test_repair_last_good_removes_only_retired_archived_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway.tee.release_archive_v2 import archive_verified_release
    from tests.test_gateway_release_archive_v2 import _release_fixture, _role_pcr0s

    _use_fixture_measurements(monkeypatch)
    gateway_root, eif_root, release_path, release = _release_fixture(
        tmp_path / "build", "a"
    )
    archive_root = tmp_path / "archive"
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )
    last_good = tmp_path / "gateway-last-good.json"
    original = {
        "schema_version": gateway_git_deploy.SCHEMA_VERSION,
        "status": "succeeded",
        "target_sha": release["commit_sha"],
        "stage": "completed",
        "sentinel": "preserved",
        "role_pcr0s": {
            **_role_pcr0s(release),
            "gateway_autoresearch": "f" * 96,
        },
    }
    last_good.write_text(json.dumps(original), encoding="utf-8")

    repaired = gateway_git_deploy.repair_last_good_role_pcr0s(
        last_good_file=last_good,
        archive_root=archive_root,
    )

    assert repaired == {**original, "role_pcr0s": _role_pcr0s(release)}
    assert json.loads(last_good.read_text(encoding="utf-8")) == repaired
    assert gateway_git_deploy.repair_last_good_role_pcr0s(
        last_good_file=last_good,
        archive_root=archive_root,
    ) == repaired

    next_gateway, next_eifs, next_release_path, next_release = _release_fixture(
        tmp_path / "next-build", "b"
    )
    archive_verified_release(
        release_manifest_path=next_release_path,
        gateway_root=next_gateway,
        eif_root=next_eifs,
        archive_root=archive_root,
        last_good_manifest_path=last_good,
    )
    for role, pcr0 in _role_pcr0s(next_release).items():
        (next_eifs / ("enclave-build-%s.json" % role)).write_text(
            json.dumps({"Measurements": {"PCR0": pcr0}}),
            encoding="utf-8",
        )
    next_plan = tmp_path / "next-plan.json"
    next_plan.write_text(
        json.dumps(
            {
                "schema_version": gateway_git_deploy.SCHEMA_VERSION,
                "target_sha": next_release["commit_sha"],
                "manifest_file": str(tmp_path / "gateway-current.json"),
                "last_good_file": str(last_good),
            }
        ),
        encoding="utf-8",
    )
    completed = gateway_git_deploy.finalize_deployment(
        plan_file=next_plan,
        status="succeeded",
        stage="completed",
        eif_root=next_eifs,
    )
    assert completed["role_pcr0s"] == _role_pcr0s(next_release)


def test_finalize_uses_exact_release_roles_and_ignores_retired_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_gateway_release_archive_v2 import _release_fixture, _role_pcr0s

    _use_fixture_measurements(monkeypatch)
    _gateway_root, eif_root, _release_path, release = _release_fixture(
        tmp_path / "build", "a"
    )
    for role, pcr0 in _role_pcr0s(release).items():
        (eif_root / ("enclave-build-%s.json" % role)).write_text(
            json.dumps({"Measurements": {"PCR0": pcr0}}),
            encoding="utf-8",
        )
    (eif_root / "enclave-build-gateway_autoresearch.json").write_text(
        json.dumps({"Measurements": {"PCR0": "f" * 96}}),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    manifest = tmp_path / "gateway-current.json"
    last_good = tmp_path / "gateway-last-good.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": gateway_git_deploy.SCHEMA_VERSION,
                "target_sha": release["commit_sha"],
                "manifest_file": str(manifest),
                "last_good_file": str(last_good),
            }
        ),
        encoding="utf-8",
    )

    completed = gateway_git_deploy.finalize_deployment(
        plan_file=plan,
        status="succeeded",
        stage="completed",
        eif_root=eif_root,
    )

    assert completed["role_pcr0s"] == _role_pcr0s(release)
    assert "gateway_autoresearch" not in completed["role_pcr0s"]


def test_repair_last_good_rejects_retained_pcr0_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway.tee.release_archive_v2 import archive_verified_release
    from tests.test_gateway_release_archive_v2 import _release_fixture, _role_pcr0s

    _use_fixture_measurements(monkeypatch)
    gateway_root, eif_root, release_path, release = _release_fixture(
        tmp_path / "build", "a"
    )
    archive_root = tmp_path / "archive"
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )
    role_pcr0s = _role_pcr0s(release)
    role_pcr0s["gateway_scoring"] = "e" * 96
    last_good = tmp_path / "gateway-last-good.json"
    original = json.dumps(
        {
            "schema_version": gateway_git_deploy.SCHEMA_VERSION,
            "status": "succeeded",
            "target_sha": release["commit_sha"],
            "role_pcr0s": {
                **role_pcr0s,
                "gateway_autoresearch": "f" * 96,
            },
        }
    )
    last_good.write_text(original, encoding="utf-8")

    with pytest.raises(
        gateway_git_deploy.GatewayGitDeployError,
        match="retained role PCR0s differ",
    ):
        gateway_git_deploy.repair_last_good_role_pcr0s(
            last_good_file=last_good,
            archive_root=archive_root,
        )

    assert last_good.read_text(encoding="utf-8") == original


def test_repair_last_good_rejects_unknown_extra_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway.tee.release_archive_v2 import archive_verified_release
    from tests.test_gateway_release_archive_v2 import _release_fixture, _role_pcr0s

    _use_fixture_measurements(monkeypatch)
    gateway_root, eif_root, release_path, release = _release_fixture(
        tmp_path / "build", "a"
    )
    archive_root = tmp_path / "archive"
    archive_verified_release(
        release_manifest_path=release_path,
        gateway_root=gateway_root,
        eif_root=eif_root,
        archive_root=archive_root,
    )
    last_good = tmp_path / "gateway-last-good.json"
    original = json.dumps(
        {
            "schema_version": gateway_git_deploy.SCHEMA_VERSION,
            "status": "succeeded",
            "target_sha": release["commit_sha"],
            "role_pcr0s": {**_role_pcr0s(release), "unknown_role": "f" * 96},
        }
    )
    last_good.write_text(original, encoding="utf-8")

    with pytest.raises(gateway_git_deploy.GatewayGitDeployError):
        gateway_git_deploy.repair_last_good_role_pcr0s(
            last_good_file=last_good,
            archive_root=archive_root,
        )
    assert last_good.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("mode", ["missing", "mismatch"])
def test_collect_role_pcr0s_rejects_missing_or_mismatched_measurement(
    tmp_path: Path, mode: str
) -> None:
    expected = {"gateway_scoring": "a" * 96}
    if mode == "mismatch":
        (tmp_path / "enclave-build-gateway_scoring.json").write_text(
            json.dumps({"Measurements": {"PCR0": "b" * 96}}),
            encoding="utf-8",
        )

    with pytest.raises(gateway_git_deploy.GatewayGitDeployError):
        gateway_git_deploy.collect_role_pcr0s(tmp_path, expected)


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


def test_prepared_and_activated_trees_are_verified_against_exact_git_blobs(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    plan, manifest, _last_good = _paths(tmp_path)
    prepared = _prepare(git_fixture, tmp_path)
    materialized = tmp_path / "materialized"
    _materialize_commit(
        git_fixture.checkout,
        prepared["target_sha"],
        materialized,
    )
    evidence_path = tmp_path / "prepared-tree.json"
    prepared_evidence = gateway_git_deploy.write_tree_verification_evidence(
        repo_root=git_fixture.checkout,
        materialized_root=materialized,
        target_sha=prepared["target_sha"],
        phase="prepared_archive",
        strict_extras=True,
        output_path=evidence_path,
    )

    gateway_git_deploy.activate_deployment(plan_file=plan)
    paired = gateway_git_deploy.record_tree_verification_pair(
        plan_file=plan,
        prepared_evidence_path=evidence_path,
        activated_root=git_fixture.checkout,
    )

    assert prepared_evidence["blob_count"] > 0
    assert prepared_evidence["strict_extras"] is True
    assert (
        paired["prepared_archive"]["blob_manifest_sha256"]
        == paired["activated_checkout"]["blob_manifest_sha256"]
    )
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(recorded["tree_verifications"]) == {
        "prepared_archive",
        "activated_checkout",
    }


def test_candidate_tree_verification_rejects_tampering_and_extra_files(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    prepared = _prepare(git_fixture, tmp_path)
    materialized = tmp_path / "materialized"
    _materialize_commit(
        git_fixture.checkout,
        prepared["target_sha"],
        materialized,
    )
    (materialized / "payload.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(
        gateway_git_deploy.GatewayGitDeployError,
        match="content mismatch",
    ):
        gateway_git_deploy.verify_materialized_tree(
            repo_root=git_fixture.checkout,
            materialized_root=materialized,
            target_sha=prepared["target_sha"],
            strict_extras=True,
        )

    _materialize_commit(
        git_fixture.checkout,
        prepared["target_sha"],
        tmp_path / "materialized-clean",
    )
    clean = tmp_path / "materialized-clean"
    runtime_cache = clean / "__pycache__"
    runtime_cache.mkdir()
    (runtime_cache / "module.cpython-311.pyc").write_bytes(b"runtime-cache")
    cache_evidence = gateway_git_deploy.verify_materialized_tree(
        repo_root=git_fixture.checkout,
        materialized_root=clean,
        target_sha=prepared["target_sha"],
        strict_extras=True,
    )
    assert cache_evidence["ignored_runtime_cache_count"] == 1

    (clean / "not-from-git.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(
        gateway_git_deploy.GatewayGitDeployError,
        match="non-Git path",
    ):
        gateway_git_deploy.verify_materialized_tree(
            repo_root=git_fixture.checkout,
            materialized_root=clean,
            target_sha=prepared["target_sha"],
            strict_extras=True,
        )


def test_tree_pair_rejects_prepared_evidence_for_another_tree(
    git_fixture: GitFixture,
    tmp_path: Path,
) -> None:
    plan, _manifest, _last_good = _paths(tmp_path)
    prepared = _prepare(git_fixture, tmp_path)
    materialized = tmp_path / "materialized"
    _materialize_commit(
        git_fixture.checkout,
        prepared["target_sha"],
        materialized,
    )
    evidence_path = tmp_path / "prepared-tree.json"
    evidence = gateway_git_deploy.write_tree_verification_evidence(
        repo_root=git_fixture.checkout,
        materialized_root=materialized,
        target_sha=prepared["target_sha"],
        phase="prepared_archive",
        strict_extras=True,
        output_path=evidence_path,
    )
    evidence["blob_manifest_sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    gateway_git_deploy.activate_deployment(plan_file=plan)

    with pytest.raises(
        gateway_git_deploy.GatewayGitDeployError,
        match="prepared and activated candidate trees differ",
    ):
        gateway_git_deploy.record_tree_verification_pair(
            plan_file=plan,
            prepared_evidence_path=evidence_path,
            activated_root=git_fixture.checkout,
        )


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
        (
            f"GITHUB_REPO_URL={git_fixture.remote}\n"
            "GITHUB_BRANCH=gateway-release\n"
            f"GATEWAY_DEPLOY_COMMIT={git_fixture.initial_sha}\n"
        ),
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


def test_prepare_cli_honors_one_invocation_rollback_pin(
    git_fixture: GitFixture,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _commit(git_fixture.source, "newer")
    _git(git_fixture.source, "push", "origin", "main")
    monkeypatch.setenv("GATEWAY_DEPLOY_COMMIT", git_fixture.initial_sha)
    plan, manifest, last_good = _paths(tmp_path)

    assert (
        gateway_git_deploy.main(
            [
                "prepare",
                "--repo-root",
                str(git_fixture.checkout),
                "--repo-url",
                str(git_fixture.remote),
                "--branch",
                "main",
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
    document = json.loads(plan.read_text(encoding="utf-8"))
    assert document["mode"] == "pinned"
    assert document["target_sha"] == git_fixture.initial_sha


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
