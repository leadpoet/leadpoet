from __future__ import annotations

import base64
import errno
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys

import pytest

from scripts import verify_installed_gateway_controller_v1 as verifier


CANDIDATE_COMMIT = "a" * 40
N_MINUS_ONE_COMMIT = next(iter(verifier.SUPPORTED_CONTROLLER_COMMITS))


def _source_candidate_commit(source_repository: Path) -> str:
    candidate = subprocess.check_output(
        ["git", "-C", str(source_repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if len(candidate) != 40:
        raise ValueError("source candidate commit is invalid")
    subprocess.run(
        [
            "git",
            "-C",
            str(source_repository),
            "merge-base",
            "--is-ancestor",
            N_MINUS_ONE_COMMIT,
            candidate,
        ],
        check=True,
    )
    return candidate


def test_stale_repository_fixture_uses_resolvable_committed_ancestry() -> None:
    source_repository = Path(__file__).resolve().parents[1]
    candidate = _source_candidate_commit(source_repository)
    assert candidate != N_MINUS_ONE_COMMIT
    for commit in (N_MINUS_ONE_COMMIT, candidate):
        subprocess.run(
            [
                "git",
                "-C",
                str(source_repository),
                "cat-file",
                "-e",
                f"{commit}^{{commit}}",
            ],
            check=True,
        )


def _controller_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    controller_commit: str = N_MINUS_ONE_COMMIT,
    host_commit: str | None = None,
):
    repository = tmp_path / "repo"
    repository.mkdir()
    controller_parent = tmp_path / "restart-controller"
    controller_root = controller_parent / "gateway"
    releases = controller_root / "releases"
    release = releases / controller_commit
    for directory in (controller_parent, controller_root, releases):
        directory.mkdir(exist_ok=True)
        directory.chmod(0o775)
    nested_directories = (
        release / "scripts",
        release / "Leadpoet",
        release / "Leadpoet/utils",
        release / "gateway",
        release / "gateway/tee",
    )
    for directory in nested_directories:
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o775)
    release.chmod(0o700)
    payloads = {
        "gw_restart.sh": b"#!/bin/bash\nexit 0\n",
        "scripts/gateway_git_deploy.py": b"HELPER = True\n",
        "Leadpoet/utils/exact_commit_restart_v2.py": b"EXACT = True\n",
        "gateway/tee/host_memory_guard_v2.py": b"GUARD = True\n",
    }
    candidate_payloads = {
        **payloads,
        "gw_restart.sh": b"#!/bin/bash\n# candidate\nexit 0\n",
    }
    installed_payloads = (
        candidate_payloads if controller_commit == CANDIDATE_COMMIT else payloads
    )
    for relative_path, payload in installed_payloads.items():
        destination = release / relative_path
        destination.write_bytes(payload)
        destination.chmod(0o700 if relative_path == "gw_restart.sh" else 0o600)
    current = controller_root / "current"
    current.symlink_to(f"releases/{controller_commit}")
    host_restart = tmp_path / "gw_restart.sh"
    selected_host_commit = host_commit or controller_commit
    selected_host_payloads = (
        candidate_payloads if selected_host_commit == CANDIDATE_COMMIT else payloads
    )
    host_restart.write_bytes(selected_host_payloads["gw_restart.sh"])
    host_restart.chmod(0o700)

    def fake_git(_repository, *arguments, binary=False):
        if arguments[:3] == (
            "for-each-ref",
            "--format=%(refname)",
            "refs/replace",
        ):
            return b"" if binary else ""
        if arguments[:2] == ("rev-parse", "--git-path"):
            return str(repository / ".git" / str(arguments[2]))
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            return b"" if binary else ""
        if arguments[0] == "ls-tree":
            relative_path = str(arguments[-1])
            git_mode = verifier.CONTROLLER_FILES[relative_path][1]
            return f"{git_mode} blob {'0' * 40}\t{relative_path}"
        if arguments[0] == "show":
            commit, relative_path = str(arguments[1]).split(":", 1)
            selected = candidate_payloads if commit == CANDIDATE_COMMIT else payloads
            result = selected[relative_path]
            return result if binary else result.decode("utf-8")
        raise AssertionError(arguments)

    monkeypatch.setattr(verifier, "_git", fake_git)
    monkeypatch.setattr(verifier, "_git_commit_exists", lambda *_args: True)
    return repository, current, host_restart, release, installed_payloads


def _installed_controller_directories(
    current: Path,
    release: Path,
) -> tuple[Path, ...]:
    controller_root = current.parent
    return (
        controller_root.parent,
        controller_root,
        controller_root / "releases",
        release,
        release / "scripts",
        release / "Leadpoet",
        release / "Leadpoet/utils",
        release / "gateway",
        release / "gateway/tee",
    )


def test_valid_bundle_allows_exact_partial_cutover_host_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repository, current, host_restart, release, _payloads = _controller_fixture(
        tmp_path,
        monkeypatch,
        controller_commit=CANDIDATE_COMMIT,
        host_commit=N_MINUS_ONE_COMMIT,
    )

    result = verifier.verify_installed_controller_bundle(
        repo_root=repository,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
    )

    assert result["controller_commit"] == CANDIDATE_COMMIT
    assert result["host_controller_commits"] == [N_MINUS_ONE_COMMIT]
    assert set(result["payloads"]) == set(verifier.CONTROLLER_FILES)
    assert all(
        stat.S_IMODE(directory.stat().st_mode) == 0o700
        for directory in _installed_controller_directories(current, release)
    )


def test_exact_supported_n_minus_one_bundle_does_not_require_candidate_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, current, host_restart, _release, _payloads = _controller_fixture(
        tmp_path,
        monkeypatch,
        controller_commit=N_MINUS_ONE_COMMIT,
        host_commit=N_MINUS_ONE_COMMIT,
    )
    monkeypatch.setattr(
        verifier,
        "_git_commit_exists",
        lambda _repository, commit: commit == N_MINUS_ONE_COMMIT,
    )

    result = verifier.verify_installed_controller_bundle(
        repo_root=repository,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
    )

    assert result["controller_commit"] == N_MINUS_ONE_COMMIT
    assert result["host_controller_commits"] == [N_MINUS_ONE_COMMIT]


def test_controller_directory_hardening_is_retry_safe_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, current, host_restart, release, _payloads = _controller_fixture(
        tmp_path,
        monkeypatch,
    )
    fake_git = verifier._git
    interrupted = False

    def interrupt_after_first_file(_repository, *arguments, binary=False):
        nonlocal interrupted
        if arguments[0] == "ls-tree" and not interrupted:
            interrupted = True
            raise verifier.InstalledGatewayControllerError("synthetic interruption")
        return fake_git(_repository, *arguments, binary=binary)

    monkeypatch.setattr(verifier, "_git", interrupt_after_first_file)
    with pytest.raises(
        verifier.InstalledGatewayControllerError,
        match="synthetic interruption",
    ):
        verifier.verify_installed_controller_bundle(
            repo_root=repository,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )

    controller_root = current.parent
    assert all(
        stat.S_IMODE(directory.stat().st_mode) == 0o700
        for directory in (
            controller_root.parent,
            controller_root,
            controller_root / "releases",
        )
    )
    assert all(
        stat.S_IMODE(directory.stat().st_mode) == 0o775
        for directory in (
            release / "scripts",
            release / "Leadpoet",
            release / "Leadpoet/utils",
            release / "gateway",
            release / "gateway/tee",
        )
    )

    monkeypatch.setattr(verifier, "_git", fake_git)
    first_retry = verifier.verify_installed_controller_bundle(
        repo_root=repository,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
    )
    second_retry = verifier.verify_installed_controller_bundle(
        repo_root=repository,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
    )

    assert first_retry["controller_commit"] == N_MINUS_ONE_COMMIT
    assert second_retry["controller_commit"] == N_MINUS_ONE_COMMIT
    assert all(
        stat.S_IMODE(directory.stat().st_mode) == 0o700
        for directory in _installed_controller_directories(current, release)
    )


def test_reviewed_nested_controller_directory_rejects_mode_777(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, current, host_restart, release, _payloads = _controller_fixture(
        tmp_path,
        monkeypatch,
    )
    (release / "scripts").chmod(0o777)

    with pytest.raises(
        verifier.InstalledGatewayControllerError,
        match="installed controller file ancestry is unsafe",
    ):
        verifier.verify_installed_controller_bundle(
            repo_root=repository,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )


def test_unreviewed_nested_group_writable_directory_remains_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, current, _host_restart, release, _payloads = _controller_fixture(
        tmp_path,
        monkeypatch,
    )
    unreviewed_directory = release / "unreviewed"
    unreviewed_directory.mkdir()
    unreviewed_directory.chmod(0o775)
    unreviewed_file = unreviewed_directory / "helper.py"
    unreviewed_file.write_bytes(b"UNREVIEWED = True\n")
    unreviewed_file.chmod(0o600)
    controller_root = current.parent
    reviewed_nested_paths = verifier._reviewed_controller_parent_paths(release)
    assert reviewed_nested_paths == frozenset(
        {
            release / "scripts",
            release / "Leadpoet",
            release / "Leadpoet/utils",
            release / "gateway",
            release / "gateway/tee",
        }
    )
    allowed_paths = frozenset(
        {
            controller_root.parent,
            controller_root,
            controller_root / "releases",
        }
    ) | reviewed_nested_paths

    with pytest.raises(
        verifier.InstalledGatewayControllerError,
        match="installed controller file ancestry is unsafe",
    ):
        verifier._read_exact_file(
            unreviewed_file,
            expected_mode=0o600,
            allowed_group_writable_paths=allowed_paths,
        )


def test_tampered_installed_helper_never_reaches_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repository, current, host_restart, release, _payloads = _controller_fixture(
        tmp_path,
        monkeypatch,
    )
    sentinel = tmp_path / "tampered-helper-executed"
    helper = release / "scripts/gateway_git_deploy.py"
    helper.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n",
        encoding="utf-8",
    )
    helper.chmod(0o600)
    execution_called = False

    def unexpected_exec(**_kwargs):
        nonlocal execution_called
        execution_called = True

    monkeypatch.setattr(verifier, "_exec_verified_helper", unexpected_exec)

    status = verifier.main(
        [
            "--repo-root",
            str(repository),
            "--controller-current",
            str(current),
            "--host-restart-path",
            str(host_restart),
            "--expected-commit",
            CANDIDATE_COMMIT,
            "--exec-helper",
            "scripts/gateway_git_deploy.py",
            "--",
            "prepare",
        ]
    )

    assert status == 2
    assert not execution_called
    assert not sentinel.exists()


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="production verifier requires Linux sealed memfd support",
)
def test_exact_operator_python_c_shape_reaches_only_verified_helper(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    source_repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["git", "clone", "-q", str(source_repository), str(repository)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "checkout", "-q", N_MINUS_ONE_COMMIT],
        check=True,
    )
    sentinel_helper = repository / "scripts/gateway_git_deploy.py"
    sentinel_helper.write_bytes(
        b"from pathlib import Path\n"
        b"import sys\n"
        b"Path(sys.argv[1]).write_text(' '.join(sys.argv[2:]), encoding='utf-8')\n"
    )
    subprocess.run(
        ["git", "-C", str(repository), "add", "scripts/gateway_git_deploy.py"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "controller",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    payloads = {
        relative_path: subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{commit}:{relative_path}"]
        )
        for relative_path in verifier.CONTROLLER_FILES
    }
    controller_root = tmp_path / "restart-controller" / "gateway"
    release = controller_root / "releases" / commit
    for directory in (
        controller_root.parent,
        controller_root,
        controller_root / "releases",
        release,
    ):
        directory.mkdir(exist_ok=True)
        directory.chmod(0o700)
    for relative_path, payload in payloads.items():
        target = release / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o700 if relative_path == "gw_restart.sh" else 0o600)
    current = controller_root / "current"
    current.symlink_to(f"releases/{commit}")
    host_restart = tmp_path / "gw_restart.sh"
    shutil.copyfile(release / "gw_restart.sh", host_restart)
    host_restart.chmod(0o700)
    sentinel = tmp_path / "verified-helper-ran"
    source_b64 = base64.b64encode(
        Path(verifier.__file__).read_bytes()
    ).decode("ascii")
    loader = (
        "import base64,sys; "
        "source=base64.b64decode(sys.stdin.buffer.read(), validate=True); "
        "exec(compile(source, '<exact-installed-controller-verifier>', 'exec'))"
    )
    command = " ".join(
        shlex.quote(value)
        for value in (
            "printf",
            "%s",
            source_b64,
        )
    ) + " | " + " ".join(
        shlex.quote(value)
        for value in (
            sys.executable,
            "-I",
            "-S",
            "-c",
            loader,
            "--repo-root",
            str(repository),
            "--controller-current",
            str(current),
            "--host-restart-path",
            str(host_restart),
            "--expected-commit",
            commit,
            "--exec-helper",
            "scripts/gateway_git_deploy.py",
            "--",
            str(sentinel),
            "prepare",
            "exact-argv",
        )
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_")
    }

    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert sentinel.read_text(encoding="utf-8") == "prepare exact-argv"


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="production verifier requires Linux sealed memfd support",
)
def test_exact_operator_shape_fetches_candidate_missing_from_deployed_repo(
    tmp_path: Path,
) -> None:
    source_repository = Path(__file__).resolve().parents[1]
    real_candidate_commit = _source_candidate_commit(source_repository)
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "clone", "--bare", "-q", str(source_repository), str(origin)],
        check=True,
    )
    for branch, commit in (
        ("deployed", N_MINUS_ONE_COMMIT),
        ("main", real_candidate_commit),
    ):
        subprocess.run(
            [
                "git",
                "--git-dir",
                str(origin),
                "cat-file",
                "-e",
                f"{commit}^{{commit}}",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "--git-dir",
                str(origin),
                "update-ref",
                f"refs/heads/{branch}",
                commit,
            ],
            check=True,
        )
    repository = tmp_path / "repo"
    remote_url = origin.as_uri()
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--no-local",
            "--single-branch",
            "--branch",
            "deployed",
            "--depth",
            "1",
            remote_url,
            str(repository),
        ],
        check=True,
    )
    absent_before = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "cat-file",
            "-e",
            f"{real_candidate_commit}^{{commit}}",
        ],
        check=False,
        capture_output=True,
    )
    assert absent_before.returncode != 0

    payloads = {
        relative_path: subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "show",
                f"{N_MINUS_ONE_COMMIT}:{relative_path}",
            ]
        )
        for relative_path in verifier.CONTROLLER_FILES
    }
    controller_root = tmp_path / "restart-controller" / "gateway"
    release = controller_root / "releases" / N_MINUS_ONE_COMMIT
    for directory in (
        controller_root.parent,
        controller_root,
        controller_root / "releases",
        release,
    ):
        directory.mkdir(exist_ok=True)
        directory.chmod(0o700)
    for relative_path, payload in payloads.items():
        target = release / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o700 if relative_path == "gw_restart.sh" else 0o600)
    current = controller_root / "current"
    current.symlink_to(f"releases/{N_MINUS_ONE_COMMIT}")
    host_restart = tmp_path / "gw_restart.sh"
    shutil.copyfile(release / "gw_restart.sh", host_restart)
    host_restart.chmod(0o700)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    plan = state / "plan.json"
    manifest = state / "manifest.json"
    last_good = state / "last-good.json"
    source_b64 = base64.b64encode(
        Path(verifier.__file__).read_bytes()
    ).decode("ascii")
    loader = (
        "import base64,sys; "
        "source=base64.b64decode(sys.stdin.buffer.read(), validate=True); "
        "exec(compile(source, '<exact-installed-controller-verifier>', 'exec'))"
    )
    command = " ".join(
        shlex.quote(value) for value in ("printf", "%s", source_b64)
    ) + " | " + " ".join(
        shlex.quote(value)
        for value in (
            sys.executable,
            "-I",
            "-S",
            "-c",
            loader,
            "--repo-root",
            str(repository),
            "--controller-current",
            str(current),
            "--host-restart-path",
            str(host_restart),
            "--expected-commit",
            real_candidate_commit,
            "--exec-helper",
            "scripts/gateway_git_deploy.py",
            "--",
            "prepare",
            "--repo-root",
            str(repository),
            "--repo-url",
            remote_url,
            "--branch",
            "main",
            "--deploy-commit",
            real_candidate_commit,
            "--plan-file",
            str(plan),
            "--manifest-file",
            str(manifest),
            "--last-good-file",
            str(last_good),
        )
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_")
    }

    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == real_candidate_commit
    assert json.loads(plan.read_text(encoding="utf-8"))["target_sha"] == (
        real_candidate_commit
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "cat-file",
            "-e",
            f"{real_candidate_commit}^{{commit}}",
        ],
        check=True,
    )


@pytest.mark.skipif(
    not hasattr(os, "memfd_create")
    or not all(
        hasattr(verifier.fcntl, name)
        for name in (
            "F_ADD_SEALS",
            "F_GET_SEALS",
            "F_SEAL_WRITE",
            "F_SEAL_GROW",
            "F_SEAL_SHRINK",
            "F_SEAL_SEAL",
        )
    ),
    reason="Linux sealed memfd support is required",
)
def test_verified_helper_exec_uses_sealed_snapshot_after_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repository, current, host_restart, release, payloads = _controller_fixture(
        tmp_path,
        monkeypatch,
    )
    bundle = verifier.verify_installed_controller_bundle(
        repo_root=repository,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
    )
    (release / "scripts/gateway_git_deploy.py").write_bytes(b"TAMPERED = True\n")
    observed: dict[str, object] = {}

    def fake_execve(executable, arguments, environment):
        descriptor = int(str(arguments[3]).rsplit("/", 1)[1])
        observed["payload"] = os.pread(descriptor, 1024 * 1024, 0)
        observed["arguments"] = list(arguments)
        observed["environment"] = dict(environment)
        with pytest.raises(OSError) as failure:
            os.write(descriptor, b"mutation")
        assert failure.value.errno in {errno.EPERM, errno.EBADF}
        raise verifier.InstalledGatewayControllerError("test exec boundary")

    monkeypatch.setattr(verifier.os, "execve", fake_execve)
    with pytest.raises(
        verifier.InstalledGatewayControllerError,
        match="test exec boundary",
    ):
        verifier._exec_verified_helper(
            bundle=bundle,
            relative_path="scripts/gateway_git_deploy.py",
            arguments=["prepare"],
        )

    assert observed["payload"] == payloads["scripts/gateway_git_deploy.py"]
    assert observed["arguments"][1:3] == ["-I", "-S"]
    assert "PYTHONPATH" not in observed["environment"]
