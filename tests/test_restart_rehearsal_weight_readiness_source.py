from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import uuid

import pytest

from tests.restart_rehearsal.weight_readiness_runner import (
    _candidate_source_identity,
)


SOURCE_PATH = Path("gateway/tee/verify_weight_submission_ready_v2.py")


def _archive_dir() -> Path:
    archive = Path("/tmp") / f"gateway-v2-preflight.{uuid.uuid4().hex}"
    archive.mkdir()
    return archive


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def candidate_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "candidate"
    source = repo / SOURCE_PATH
    source.parent.mkdir(parents=True)
    source.write_text("CANDIDATE = True\n", encoding="utf-8")
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "rehearsal@example.invalid", cwd=repo)
    _git("config", "user.name", "Restart Rehearsal", cwd=repo)
    _git("add", SOURCE_PATH.as_posix(), cwd=repo)
    _git("commit", "-qm", "candidate", cwd=repo)
    return repo, _git("rev-parse", "HEAD", cwd=repo)


def test_candidate_checkout_source_is_bound_to_exact_git_blob(
    candidate_repo: tuple[Path, str],
) -> None:
    repo, candidate_sha = candidate_repo

    identity = _candidate_source_identity(
        repo / SOURCE_PATH,
        expected_root=repo,
        candidate_sha=candidate_sha,
    )

    assert identity["source_kind"] == "candidate_checkout"
    assert identity["source_commit"] == candidate_sha
    assert identity["source_git_path"] == SOURCE_PATH.as_posix()


def test_pre_activation_archive_is_bound_to_exact_candidate_git_blob(
    candidate_repo: tuple[Path, str],
) -> None:
    repo, candidate_sha = candidate_repo
    archive = _archive_dir()
    try:
        archived_source = archive / SOURCE_PATH
        archived_source.parent.mkdir(parents=True)
        shutil.copyfile(repo / SOURCE_PATH, archived_source)

        identity = _candidate_source_identity(
            archived_source,
            expected_root=repo,
            candidate_sha=candidate_sha,
        )

        assert identity["source_kind"] == "candidate_archive"
        assert identity["source_commit"] == candidate_sha
        assert identity["source_git_path"] == SOURCE_PATH.as_posix()
    finally:
        shutil.rmtree(archive)


def test_tampered_pre_activation_archive_is_rejected(
    candidate_repo: tuple[Path, str],
) -> None:
    repo, candidate_sha = candidate_repo
    archive = _archive_dir()
    try:
        archived_source = archive / SOURCE_PATH
        archived_source.parent.mkdir(parents=True)
        archived_source.write_text("CANDIDATE = False\n", encoding="utf-8")

        with pytest.raises(
            RuntimeError,
            match="source bytes differ from the candidate commit",
        ):
            _candidate_source_identity(
                archived_source,
                expected_root=repo,
                candidate_sha=candidate_sha,
            )
    finally:
        shutil.rmtree(archive)


def test_unrecognized_archive_path_is_rejected(
    candidate_repo: tuple[Path, str],
    tmp_path: Path,
) -> None:
    repo, candidate_sha = candidate_repo
    unrecognized = tmp_path / "unrecognized" / SOURCE_PATH
    unrecognized.parent.mkdir(parents=True)
    shutil.copyfile(repo / SOURCE_PATH, unrecognized)

    with pytest.raises(RuntimeError, match="recognized candidate"):
        _candidate_source_identity(
            unrecognized,
            expected_root=repo,
            candidate_sha=candidate_sha,
        )
