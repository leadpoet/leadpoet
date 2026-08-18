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
    release_file = tmp_path / "holder.release"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_file),
        "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "1",
    }
    holder_env = {
        **env,
        "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "5",
    }
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                "set -e; "
                f'. "{LOCK_HELPER}"; '
                "leadpoet_acquire_docker_operation_lock_v2; "
                f'touch "{ready_file}"; '
                f'while [ ! -e "{release_file}" ]; do sleep 0.02; done'
            ),
        ],
        env=holder_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            not ready_file.exists()
            and holder.poll() is None
            and time.monotonic() < deadline
        ):
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
        release_file.touch()
        holder_stdout, holder_stderr = holder.communicate(timeout=4)

    assert holder.returncode == 0, (holder_stdout, holder_stderr)

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


def test_shell_and_python_canonicalize_symlinked_turnstile_paths(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_flock(bin_dir)
    resource_target = tmp_path / "resource.lock"
    resource_target.touch()
    resource_alias = tmp_path / "resource-alias.lock"
    resource_alias.symlink_to(resource_target)
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'. "{LOCK_HELPER}"; '
                "leadpoet_acquire_docker_operation_lock_v2; "
                'printf "RESOURCE=%s\\nADMISSION=%s\\n" '
                '"$LEADPOET_DOCKER_OPERATION_LOCK_FILE" '
                '"$LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE"; '
                "leadpoet_release_docker_operation_lock_v2"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(resource_alias),
            "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "2",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    canonical_resource = resource_target.resolve()
    assert result.returncode == 0, result.stderr
    assert f"RESOURCE={canonical_resource}" in result.stdout
    assert f"ADMISSION={canonical_resource}.admission" in result.stdout


def test_post_activation_reacquires_lock_missing_from_older_launcher(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_flock(bin_dir)
    lock_file = tmp_path / "docker-operation.lock"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_file),
        "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "1",
        "LEADPOET_DOCKER_OPERATION_LOCK_HELD": "1",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'. "{LOCK_HELPER}"; '
                "leadpoet_ensure_post_activation_docker_operation_lock_v2; "
                'test "$(readlink /proc/$$/fd/7)" = '
                '"$LEADPOET_DOCKER_OPERATION_LOCK_FILE"; '
                "leadpoet_release_docker_operation_lock_v2"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Post-activation Docker lock was not inherited" in result.stdout


def test_post_activation_rejects_unexpected_fd_seven(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    other_file = tmp_path / "unrelated.lock"
    lock_file = tmp_path / "docker-operation.lock"
    _write_executable(
        bin_dir / "readlink",
        f'#!/bin/sh\nprintf "%s\\n" "{other_file}"\n',
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'. "{LOCK_HELPER}"; '
                "leadpoet_ensure_post_activation_docker_operation_lock_v2"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_file),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "descriptor targets an unexpected file" in result.stderr


def test_post_activation_reacquire_respects_competing_lock(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_flock(bin_dir)
    lock_file = tmp_path / "docker-operation.lock"
    ready_file = tmp_path / "holder.ready"
    release_file = tmp_path / "holder.release"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_file),
        "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "1",
    }
    holder_env = {
        **env,
        "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "5",
    }
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                "set -e; "
                f'. "{LOCK_HELPER}"; '
                "leadpoet_acquire_docker_operation_lock_v2; "
                f'touch "{ready_file}"; '
                f'while [ ! -e "{release_file}" ]; do sleep 0.02; done'
            ),
        ],
        env=holder_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            not ready_file.exists()
            and holder.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert ready_file.exists(), holder.communicate(timeout=1)

        blocked = subprocess.run(
            [
                "bash",
                "-c",
                (
                    f'. "{LOCK_HELPER}"; '
                    "leadpoet_ensure_post_activation_docker_operation_lock_v2"
                ),
            ],
            env={
                **env,
                "LEADPOET_DOCKER_OPERATION_LOCK_HELD": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert blocked.returncode != 0
        assert "timed out waiting" in blocked.stderr
    finally:
        release_file.touch()
        holder_stdout, holder_stderr = holder.communicate(timeout=4)

    assert holder.returncode == 0, (holder_stdout, holder_stderr)


def test_post_activation_accepts_exact_inherited_lock(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    lock_file = tmp_path / "docker-operation.lock"
    _write_executable(
        bin_dir / "readlink",
        f'#!/bin/sh\nprintf "%s\\n" "{lock_file}"\n',
    )
    _write_executable(
        bin_dir / "flock",
        "#!/bin/sh\nexit 99\n",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'. "{LOCK_HELPER}"; '
                "leadpoet_ensure_post_activation_docker_operation_lock_v2"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_file),
            "LEADPOET_DOCKER_OPERATION_LOCK_HELD": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
