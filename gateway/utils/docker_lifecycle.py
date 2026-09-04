"""Small shared lock for non-destructive Docker work on the gateway host."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import time
from typing import Iterator


LOCK_FILE_ENV = "LEADPOET_DOCKER_OPERATION_LOCK_FILE"
ADMISSION_FILE_ENV = "LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE"
DEFAULT_LOCK_FILE = "/home/ec2-user/.config/leadpoet/docker-operation-v2.lock"
POLL_SECONDS = 0.25


class DockerLifecycleError(RuntimeError):
    """Docker lifecycle coordination failed within its time limit."""


def _path(name: str, default: str) -> Path:
    value = os.path.realpath(os.path.expanduser(str(os.getenv(name) or default)))
    if not value or not os.path.isabs(value):
        raise DockerLifecycleError(f"{name} must be an absolute path")
    return Path(value)


def _lock_until(fd: int, operation: int, *, deadline: float) -> None:
    while True:
        try:
            fcntl.flock(fd, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DockerLifecycleError("Docker lifecycle lock timed out")
            time.sleep(min(POLL_SECONDS, remaining))


def _wait_for_docker(*, deadline: float) -> None:
    last_error = ""
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=max(0.1, min(10.0, remaining)),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = type(exc).__name__
        else:
            if result.returncode == 0:
                return
            last_error = (result.stderr or result.stdout or "")[-200:]
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise DockerLifecycleError("Docker daemon is unavailable: " + last_error)


@contextmanager
def shared_docker_lifecycle(*, timeout_seconds: float) -> Iterator[None]:
    """Coordinate a public run with the host's exclusive Docker maintenance."""

    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise DockerLifecycleError("Docker lifecycle timeout must be positive")
    deadline = time.monotonic() + timeout
    lock_path = _path(LOCK_FILE_ENV, DEFAULT_LOCK_FILE)
    admission_path = _path(
        ADMISSION_FILE_ENV,
        str(lock_path) + ".admission",
    )
    if lock_path == admission_path:
        raise DockerLifecycleError("Docker lock paths must differ")
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    admission_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    admission_fd = os.open(admission_path, os.O_CREAT | os.O_RDWR, 0o600)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    admission_held = False
    shared_held = False
    try:
        _lock_until(admission_fd, fcntl.LOCK_EX, deadline=deadline)
        admission_held = True
        _lock_until(lock_fd, fcntl.LOCK_SH, deadline=deadline)
        shared_held = True
        fcntl.flock(admission_fd, fcntl.LOCK_UN)
        admission_held = False
        _wait_for_docker(deadline=deadline)
        yield
    except DockerLifecycleError:
        raise
    except OSError as exc:
        raise DockerLifecycleError("Docker lifecycle lock is unavailable") from exc
    finally:
        if admission_held:
            fcntl.flock(admission_fd, fcntl.LOCK_UN)
        if shared_held:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(admission_fd)
        os.close(lock_fd)
