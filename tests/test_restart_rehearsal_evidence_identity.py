from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from tests.restart_rehearsal.verify_evidence import (
    _verify_production_identity,
)


SOURCE_PATH = Path("scripts/gateway_git_deploy.py")


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
