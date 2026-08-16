"""Shared-host Docker lifecycle coordination for Research Lab work.

Long-lived model runs and non-destructive image operations take a shared
``flock``.  Restart, release, PCR0, and storage-reclaim paths take the matching
exclusive lock through ``validator_tee/scripts/docker_operation_lock_v2.sh``.
The lock is held by the worker thread itself, so cancelling an
``asyncio.to_thread`` caller cannot release it while Docker is still running.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Iterator, Mapping, Sequence


DOCKER_OPERATION_LOCK_FILE_ENV = "LEADPOET_DOCKER_OPERATION_LOCK_FILE"
DOCKER_OPERATION_ADMISSION_LOCK_FILE_ENV = (
    "LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE"
)
DOCKER_OPERATION_LOCK_TIMEOUT_ENV = "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS"
DOCKER_DAEMON_READY_TIMEOUT_ENV = "LEADPOET_DOCKER_DAEMON_READY_TIMEOUT_SECONDS"
DEFAULT_DOCKER_OPERATION_LOCK_FILE = (
    "/home/ec2-user/.config/leadpoet/docker-operation-v2.lock"
)
DEFAULT_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS = 3600
DEFAULT_DOCKER_DAEMON_READY_TIMEOUT_SECONDS = 120
DOCKER_OPERATION_LOCK_POLL_SECONDS = 0.25
DOCKER_DAEMON_READY_POLL_SECONDS = 1.0


class DockerOperationLockError(RuntimeError):
    """A bounded shared-lock or Docker-readiness failure."""


_THREAD_STATE = threading.local()


def _positive_integer_environment(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default)) or "").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise DockerOperationLockError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise DockerOperationLockError(f"{name} must be a positive integer")
    return value


def docker_operation_lock_path() -> Path:
    raw_path = str(
        os.getenv(
            DOCKER_OPERATION_LOCK_FILE_ENV,
            DEFAULT_DOCKER_OPERATION_LOCK_FILE,
        )
        or ""
    ).strip()
    expanded = os.path.expanduser(raw_path)
    if not expanded or not os.path.isabs(expanded):
        raise DockerOperationLockError(
            f"{DOCKER_OPERATION_LOCK_FILE_ENV} must be an absolute path"
        )
    return Path(os.path.realpath(expanded))


def docker_operation_admission_lock_path(
    operation_lock_path: Path | None = None,
) -> Path:
    """Return the writer-admission turnstile paired with the resource lock."""

    resource_path = operation_lock_path or docker_operation_lock_path()
    configured = str(
        os.getenv(DOCKER_OPERATION_ADMISSION_LOCK_FILE_ENV, "") or ""
    ).strip()
    raw_path = configured or f"{resource_path}.admission"
    expanded = os.path.expanduser(raw_path)
    if not expanded or not os.path.isabs(expanded):
        raise DockerOperationLockError(
            f"{DOCKER_OPERATION_ADMISSION_LOCK_FILE_ENV} must be an absolute path"
        )
    admission_path = Path(os.path.realpath(expanded))
    if admission_path == resource_path:
        raise DockerOperationLockError(
            "Docker operation admission and resource locks must differ"
        )
    return admission_path


def _acquire_file_lock_until(
    descriptor: int,
    operation: int,
    *,
    deadline_monotonic: float,
    timeout_message: str,
) -> None:
    while True:
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise DockerOperationLockError(timeout_message)
            time.sleep(min(DOCKER_OPERATION_LOCK_POLL_SECONDS, remaining))


def _bounded_deadline(
    *,
    timeout_seconds: float,
    deadline_monotonic: float | None,
) -> float:
    try:
        requested = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise DockerOperationLockError(
            "Docker operation timeout must be positive"
        ) from exc
    if not math.isfinite(requested) or requested <= 0:
        raise DockerOperationLockError("Docker operation timeout must be positive")
    configured = float(
        _positive_integer_environment(
            DOCKER_OPERATION_LOCK_TIMEOUT_ENV,
            DEFAULT_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS,
        )
    )
    deadline = time.monotonic() + min(requested, configured)
    if deadline_monotonic is not None:
        try:
            external_deadline = float(deadline_monotonic)
        except (TypeError, ValueError) as exc:
            raise DockerOperationLockError(
                "Docker operation deadline must be finite"
            ) from exc
        if not math.isfinite(external_deadline):
            raise DockerOperationLockError("Docker operation deadline must be finite")
        deadline = min(deadline, external_deadline)
    if deadline <= time.monotonic():
        raise DockerOperationLockError(
            "Docker operation deadline was already exhausted"
        )
    return deadline


def _docker_ready_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    if environment is None:
        return dict(os.environ)
    return {str(name): str(value) for name, value in environment.items()}


def wait_for_docker_daemon_ready(
    *,
    docker_executable: str = "docker",
    environment: Mapping[str, str] | None = None,
    deadline_monotonic: float,
) -> None:
    """Poll ``docker info`` while the lifecycle lock is already held."""

    ready_timeout = float(
        _positive_integer_environment(
            DOCKER_DAEMON_READY_TIMEOUT_ENV,
            DEFAULT_DOCKER_DAEMON_READY_TIMEOUT_SECONDS,
        )
    )
    deadline = min(
        float(deadline_monotonic),
        time.monotonic() + ready_timeout,
    )
    executable = str(docker_executable or "").strip()
    if not executable:
        raise DockerOperationLockError("Docker executable is required")
    last_error = ""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = f": {last_error}" if last_error else ""
            raise DockerOperationLockError(
                "Docker daemon did not become ready after maintenance" + detail
            )
        try:
            completed = subprocess.run(
                [executable, "info"],
                text=True,
                capture_output=True,
                timeout=max(0.1, min(10.0, remaining)),
                env=_docker_ready_environment(environment),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = str(exc)[:500]
        else:
            if completed.returncode == 0:
                return
            last_error = " ".join(
                (
                    str(completed.stderr or ""),
                    str(completed.stdout or ""),
                )
            ).strip()[-500:]
        time.sleep(min(DOCKER_DAEMON_READY_POLL_SECONDS, remaining))


@contextmanager
def shared_docker_operation_lock(
    *,
    timeout_seconds: float,
    docker_executable: str = "docker",
    environment: Mapping[str, str] | None = None,
    deadline_monotonic: float | None = None,
    require_daemon_ready: bool = True,
) -> Iterator[None]:
    """Hold shared access across one complete non-destructive lifecycle.

    Multiple scoring/model lifecycles may coexist.  Exclusive maintenance
    waits until every shared holder completes, and a newly entering lifecycle
    cannot pass while maintenance owns the lock.
    """

    lock_path = docker_operation_lock_path()
    admission_path = docker_operation_admission_lock_path(lock_path)
    deadline = _bounded_deadline(
        timeout_seconds=timeout_seconds,
        deadline_monotonic=deadline_monotonic,
    )
    depth = int(getattr(_THREAD_STATE, "depth", 0) or 0)
    if depth:
        if getattr(_THREAD_STATE, "path", None) != str(lock_path) or getattr(
            _THREAD_STATE, "admission_path", None
        ) != str(admission_path):
            raise DockerOperationLockError(
                "nested Docker operation lock path differs from its owner"
            )
        _THREAD_STATE.depth = depth + 1
        try:
            yield
        finally:
            _THREAD_STATE.depth = depth
        return

    admission_fd = -1
    lock_fd = -1
    admission_lock_acquired = False
    file_lock_acquired = False
    try:
        lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(lock_path.parent, 0o700)
        admission_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(admission_path.parent, 0o700)
        admission_fd = os.open(admission_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.chmod(admission_path, 0o600)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.chmod(lock_path, 0o600)
        _acquire_file_lock_until(
            admission_fd,
            fcntl.LOCK_EX,
            deadline_monotonic=deadline,
            timeout_message=("timed out waiting for Docker lifecycle admission access"),
        )
        admission_lock_acquired = True
        _acquire_file_lock_until(
            lock_fd,
            fcntl.LOCK_SH,
            deadline_monotonic=deadline,
            timeout_message="timed out waiting for shared Docker lifecycle access",
        )
        file_lock_acquired = True
        # A queued writer owns the admission lock before waiting for the
        # resource lock. Releasing admission only after our shared resource
        # lock is held prevents later readers from bypassing that writer.
        fcntl.flock(admission_fd, fcntl.LOCK_UN)
        admission_lock_acquired = False
        os.close(admission_fd)
        admission_fd = -1
        if require_daemon_ready:
            wait_for_docker_daemon_ready(
                docker_executable=docker_executable,
                environment=environment,
                deadline_monotonic=deadline,
            )
        _THREAD_STATE.depth = 1
        _THREAD_STATE.path = str(lock_path)
        _THREAD_STATE.admission_path = str(admission_path)
        _THREAD_STATE.fd = lock_fd
        try:
            yield
        finally:
            _THREAD_STATE.depth = 0
            for attribute in ("path", "admission_path", "fd"):
                try:
                    delattr(_THREAD_STATE, attribute)
                except AttributeError:
                    pass
    except DockerOperationLockError:
        raise
    except OSError as exc:
        raise DockerOperationLockError("Docker operation lock is unavailable") from exc
    finally:
        if admission_lock_acquired:
            fcntl.flock(admission_fd, fcntl.LOCK_UN)
        if file_lock_acquired:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        if admission_fd >= 0:
            os.close(admission_fd)
        if lock_fd >= 0:
            os.close(lock_fd)


def shared_docker_operation_source_paths() -> Sequence[str]:
    """Stable source inventory used by exact-transition rehearsal."""

    return (
        "research_lab/docker_operation_lock_v2.py",
        "validator_tee/scripts/docker_operation_lock_v2.sh",
        "gateway/utils/pcr0_builder.py",
    )
