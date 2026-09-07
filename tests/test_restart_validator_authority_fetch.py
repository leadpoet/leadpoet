"""A gateway-only restart must resolve authority without moving the validator."""

import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/restart_attested_release_local.sh"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repositories(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q")
    git(origin, "config", "user.email", "test@example.invalid")
    git(origin, "config", "user.name", "Test")
    git(origin, "commit", "-q", "--allow-empty", "-m", "running release")
    base = git(origin, "rev-parse", "HEAD")
    validator = tmp_path / "validator"
    subprocess.run(["git", "clone", "-q", str(origin), str(validator)], check=True)
    (validator / "operator-notes.txt").write_text("preserve existing work\n")
    git(origin, "commit", "-q", "--allow-empty", "-m", "new authority")
    authority = git(origin, "rev-parse", "HEAD")
    return validator, base, authority


def run_preparation(validator, authority):
    function = SCRIPT.read_text().split("prepare_running_validator_release_requirements() {", 1)[1]
    # Execute the actual remote Git preparation before the first runtime check.
    prefix = function.split("     cd '$VALIDATOR_REPO_ROOT'", 1)[1].split('     test \\"', 1)[0]
    command = prefix.replace("$branch_commit", authority).replace("\\\\\n", "\\\n")
    return subprocess.run(
        ["bash", "-ec", command + "\nprintf 'lineage-ready\\n'"],
        cwd=validator, text=True, capture_output=True,
    )


def assert_preserved(validator, base):
    assert git(validator, "rev-parse", "HEAD") == base
    assert (validator / "operator-notes.txt").read_text() == "preserve existing work\n"
    assert git(validator, "status", "--porcelain") == "?? operator-notes.txt"


def test_missing_authority_is_fetched_without_changing_running_checkout(repositories):
    validator, base, authority = repositories
    result = run_preparation(validator, authority)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lineage-ready"
    assert git(validator, "cat-file", "-t", authority) == "commit"
    assert_preserved(validator, base)


def test_present_authority_does_not_need_remote_access(repositories):
    validator, base, authority = repositories
    git(validator, "fetch", "-q", "origin", authority)
    git(validator, "remote", "set-url", "origin", "/nonexistent-authority-remote")
    result = run_preparation(validator, authority)
    assert result.returncode == 0, result.stderr
    assert_preserved(validator, base)


def test_unavailable_authority_stops_before_lineage_preparation(repositories):
    validator, base, authority = repositories
    git(validator, "remote", "set-url", "origin", "/nonexistent-authority-remote")
    result = run_preparation(validator, authority)
    assert result.returncode != 0
    assert "lineage-ready" not in result.stdout
    assert_preserved(validator, base)
