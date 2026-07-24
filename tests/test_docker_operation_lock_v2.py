from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
LOCK_HELPER = ROOT / "validator_tee" / "scripts" / "docker_operation_lock_v2.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _fake_flock(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "flock",
        """#!/usr/bin/env python3
import fcntl
import sys
import time

args = sys.argv[1:]
if args[0] == "-u":
    fcntl.flock(int(args[1]), fcntl.LOCK_UN)
    raise SystemExit(0)
if args[0] != "-w":
    raise SystemExit("unsupported flock invocation")

timeout = float(args[1])
fd = int(args[2])
deadline = time.monotonic() + timeout
while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(0)
    except BlockingIOError:
        if time.monotonic() >= deadline:
            raise SystemExit(1)
        time.sleep(0.02)
""",
    )


def test_shell_lock_excludes_competing_docker_operation(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_flock(bin_dir)
    lock_file = tmp_path / "docker-operation.lock"
    ready_file = tmp_path / "holder.ready"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_file),
        "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "1",
    }
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                f'. "{LOCK_HELPER}"; '
                "leadpoet_acquire_docker_operation_lock_v2; "
                f'touch "{ready_file}"; '
                "sleep 2"
            ),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 2
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready_file.exists(), holder.communicate(timeout=1)

        blocked = subprocess.run(
            [
                "bash",
                "-c",
                f'. "{LOCK_HELPER}"; leadpoet_acquire_docker_operation_lock_v2',
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert blocked.returncode != 0
        assert "timed out waiting" in blocked.stderr
    finally:
        holder.wait(timeout=4)

    acquired_after_release = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'. "{LOCK_HELPER}"; '
                "leadpoet_acquire_docker_operation_lock_v2; "
                "leadpoet_release_docker_operation_lock_v2"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert acquired_after_release.returncode == 0
