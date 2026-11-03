import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/select_attested_release_source.sh"


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def history(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Release test")
    _git(root, "config", "user.email", "release-test@example.invalid")
    _git(root, "commit", "--allow-empty", "-qm", "main ancestor")
    ancestor = _git(root, "rev-parse", "HEAD")
    _git(root, "commit", "--allow-empty", "-qm", "current main")
    current = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/main", current)
    _git(root, "checkout", "-qb", "unmerged", ancestor)
    _git(root, "commit", "--allow-empty", "-qm", "unmerged source")
    unmerged = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", current)
    return root, ancestor, current, unmerged


def _select(history, *, reproduce="", pcr0=""):
    root, _, current, _ = history
    output = root / "output"
    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=root,
        env={
            **os.environ, "GITHUB_SHA": current, "GITHUB_OUTPUT": str(output),
            "REPRODUCE_COMMIT": reproduce, "EXPECTED_PCR0": pcr0,
        },
        capture_output=True, text=True, check=False,
    )
    return result, output.read_text() if output.exists() else ""


def test_normal_release_selects_only_current_main(history):
    result, output = _select(history)
    assert result.returncode == 0, result.stderr
    assert output == f"build_commit={history[2]}\nmode=publish\n"


def test_historical_main_source_is_reproduction_only(history):
    result, output = _select(history, reproduce=history[1], pcr0="a" * 96)
    assert result.returncode == 0, result.stderr
    assert output == f"build_commit={history[1]}\nmode=reproduce\n"


def test_same_commit_reproduction_cannot_become_publication(history):
    result, output = _select(history, reproduce=history[2])
    assert result.returncode == 0
    assert "mode=reproduce" in output


def test_unmerged_source_cannot_enter_a_reproduction(history):
    result, output = _select(history, reproduce=history[3])
    assert result.returncode != 0
    assert output == ""


@pytest.mark.parametrize("value", ["main", "HEAD", "a" * 39, "a" * 41, "A" * 40, "a\nmode=publish"])
def test_source_input_must_be_one_full_lowercase_commit(history, value):
    result, output = _select(history, reproduce=value)
    assert result.returncode != 0
    assert output == ""


def test_shell_text_input_is_never_executed(history):
    marker = history[0] / "executed"
    result, _ = _select(history, reproduce=f"$(touch {marker})")
    assert result.returncode != 0
    assert not marker.exists()


@pytest.mark.parametrize("value", ["a" * 95, "A" * 96, "0" * 96, "a\nmode=publish"])
def test_expected_measurement_requires_valid_reproduction_input(history, value):
    result, output = _select(history, reproduce=history[1], pcr0=value)
    assert result.returncode != 0
    assert output == ""


def test_measurement_input_cannot_change_normal_release(history):
    result, output = _select(history, pcr0="a" * 96)
    assert result.returncode != 0
    assert output == ""


@pytest.mark.parametrize(
    ("domain", "runner_environment", "account"),
    [("gateway", "self-hosted", ""), ("validator", "self-hosted", "493765492819")],
)
def test_builder_rejects_production_execution_before_toolchain_or_docker(
    tmp_path, domain, runner_environment, account
):
    uname = tmp_path / "uname"
    uname.write_text("#!/bin/sh\nprintf 'x86_64\\n'\n", encoding="utf-8")
    uname.chmod(0o700)
    checked_account = tmp_path / "checked-account"
    binary = tmp_path / "aws"
    binary.write_text(
        f"#!/bin/sh\ntouch '{checked_account}'\nprintf '%s\\n' '{account}'\n",
        encoding="utf-8",
    )
    binary.chmod(0o700)
    result = subprocess.run(
        ["bash", str(ROOT / ".github/scripts/build_attested_release.sh")],
        env={
            **os.environ, "GATEWAY_V2_BUILDER_DOMAIN": domain,
            "RUNNER_ENVIRONMENT": runner_environment,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        }, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "toolchain" not in result.stdout
    assert checked_account.exists() == (domain == "validator")
