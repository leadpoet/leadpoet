from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "validator_tee" / "scripts" / "docker_build_retry_v2.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_retry(tmp_path: Path, *, transient: bool) -> tuple[int, int]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    count_path = tmp_path / "attempts"
    _write_executable(
        bin_dir / "docker",
        """#!/bin/bash
if [ "${1:-}" = "info" ]; then
  exit 0
fi
exit 2
""",
    )
    if transient:
        body = """#!/bin/bash
count="$(cat "$COUNT_PATH" 2>/dev/null || printf 0)"
count=$((count + 1))
printf '%s' "$count" > "$COUNT_PATH"
if [ "$count" -eq 1 ]; then
  echo 'rpc error: code = Unavailable desc = error reading from server: EOF' >&2
  exit 1
fi
echo success
"""
    else:
        body = """#!/bin/bash
count="$(cat "$COUNT_PATH" 2>/dev/null || printf 0)"
count=$((count + 1))
printf '%s' "$count" > "$COUNT_PATH"
echo 'Dockerfile parse error' >&2
exit 1
"""
    builder = bin_dir / "builder"
    _write_executable(builder, body)
    command = (
        f'. "{HELPER}"; '
        f'leadpoet_run_docker_build_with_retry_v2 test "{builder}"'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "COUNT_PATH": str(count_path),
            "VALIDATOR_DOCKER_BUILD_MAX_ATTEMPTS": "3",
            "VALIDATOR_DOCKER_BUILD_RETRY_BACKOFF_SECONDS": "0",
        },
    )
    return result.returncode, int(count_path.read_text(encoding="utf-8"))


def test_transient_docker_transport_failure_is_retried(tmp_path: Path) -> None:
    returncode, attempts = _run_retry(tmp_path, transient=True)

    assert returncode == 0
    assert attempts == 2


def test_deterministic_build_failure_is_not_retried(tmp_path: Path) -> None:
    returncode, attempts = _run_retry(tmp_path, transient=False)

    assert returncode != 0
    assert attempts == 1
