from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILES = (
    ROOT / "validator_tee" / "Dockerfile.base",
    ROOT / "validator_tee" / "Dockerfile.drand-builder",
)


def _first_run_shell(dockerfile: Path) -> str:
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("RUN "))
    command_parts = [lines[start][len("RUN ") :]]
    index = start
    while command_parts[-1].rstrip().endswith("\\"):
        command_parts[-1] = command_parts[-1].rstrip()[:-1]
        index += 1
        command_parts.append(lines[index].strip())
    return " ".join(command_parts)


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _run_install(
    tmp_path: Path,
    dockerfile: Path,
    *,
    yum_failures: int,
    provide_python: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    yum_state = tmp_path / "yum-attempts"
    sleep_log = tmp_path / "sleep-delays"
    _write_executable(
        fake_bin / "yum",
        """
if [ "${1:-}" = "install" ]; then
    attempt=0
    if [ -f "$FAKE_YUM_STATE" ]; then
        IFS= read -r attempt < "$FAKE_YUM_STATE"
    fi
    attempt=$((attempt + 1))
    printf '%s\n' "$attempt" > "$FAKE_YUM_STATE"
    if [ "$attempt" -le "$FAKE_YUM_FAILURES" ]; then
        exit 42
    fi
    exit 0
fi
if [ "${1:-}" = "clean" ]; then
    exit 0
fi
exit 97
""",
    )
    _write_executable(
        fake_bin / "sleep",
        "printf '%s\\n' \"${1:-}\" >> \"$FAKE_SLEEP_LOG\"\n",
    )
    for command in ("chmod", "find", "mkdir", "rm"):
        _write_executable(fake_bin / command, "exit 0\n")
    if provide_python:
        _write_executable(fake_bin / "python3", "exit 0\n")

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(fake_bin),
            "FAKE_YUM_FAILURES": str(yum_failures),
            "FAKE_YUM_STATE": str(yum_state),
            "FAKE_SLEEP_LOG": str(sleep_log),
        }
    )
    result = subprocess.run(
        ["/bin/sh", "-c", _first_run_shell(dockerfile)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    return result, yum_state, sleep_log


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_pinned_yum_install_retries_transient_failures(
    tmp_path: Path, dockerfile: Path
) -> None:
    result, yum_state, sleep_log = _run_install(
        tmp_path, dockerfile, yum_failures=2
    )

    assert result.returncode == 0, result.stderr
    assert yum_state.read_text(encoding="utf-8").splitlines() == ["3"]
    assert sleep_log.read_text(encoding="utf-8").splitlines() == ["5", "10"]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_pinned_yum_install_exhaustion_fails_after_three_attempts(
    tmp_path: Path, dockerfile: Path
) -> None:
    result, yum_state, sleep_log = _run_install(
        tmp_path, dockerfile, yum_failures=3
    )

    assert result.returncode != 0
    assert yum_state.read_text(encoding="utf-8").splitlines() == ["3"]
    assert sleep_log.read_text(encoding="utf-8").splitlines() == ["5", "10"]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_pinned_yum_install_requires_python_postcondition(
    tmp_path: Path, dockerfile: Path
) -> None:
    result, yum_state, _sleep_log = _run_install(
        tmp_path,
        dockerfile,
        yum_failures=0,
        provide_python=False,
    )

    assert result.returncode != 0
    assert yum_state.read_text(encoding="utf-8").splitlines() == ["1"]


def test_pinned_yum_install_keeps_versions_and_scopes_tolerated_cleanup() -> None:
    expected_packages = {
        "Dockerfile.base": {
            "python3-3.7.16-1.amzn2.0.24",
            "python3-libs-3.7.16-1.amzn2.0.24",
            "python3-pip-20.2.2-1.amzn2.0.15",
            "python3-setuptools-49.1.3-1.amzn2.0.6",
        },
        "Dockerfile.drand-builder": {
            "gcc-7.3.1-18.amzn2",
            "gcc-c++-7.3.1-18.amzn2",
            "git-2.47.3-1.amzn2.0.1",
            "gzip-1.5-10.amzn2.0.1",
            "python3-3.7.16-1.amzn2.0.24",
            "tar-1.26-35.amzn2.0.4",
        },
    }

    for dockerfile in DOCKERFILES:
        command = _first_run_shell(dockerfile)
        install_arguments = (
            command.split("yum install -y", 1)[1].split("; then", 1)[0].split()
        )
        assert set(install_arguments) == expected_packages[dockerfile.name]
        assert "set -eu" in command
        assert "for yum_attempt in 1 2 3" in command
        assert '[ "$yum_install_succeeded" -eq 1 ]' in command
        assert "assert sys.version_info[:3] == (3, 7, 16)" in command
        assert "|| true" not in command
        assert "|| :" not in command
