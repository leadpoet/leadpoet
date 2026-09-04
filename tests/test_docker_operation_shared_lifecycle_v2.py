from __future__ import annotations

import asyncio
import fcntl
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest

from research_lab import docker_operation_lock_v2 as docker_lock


@pytest.mark.parametrize("timeout_seconds", (0, -1, float("nan"), float("inf")))
def test_shared_lifecycle_rejects_non_positive_or_non_finite_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout_seconds: float,
) -> None:
    monkeypatch.setenv(
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        str(tmp_path / "docker-operation.lock"),
    )
    with pytest.raises(docker_lock.DockerOperationLockError, match="positive"):
        with docker_lock.shared_docker_operation_lock(
            timeout_seconds=timeout_seconds
        ):
            pass


def test_queued_writer_turnstile_blocks_late_shared_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_file = tmp_path / "docker-operation.lock"
    admission_file = docker_lock.docker_operation_admission_lock_path(lock_file)
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_FILE", str(lock_file))
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS", "5")
    monkeypatch.setattr(
        docker_lock.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["docker", "info"], returncode=0
        ),
    )
    first_reader_entered = threading.Event()
    release_first_reader = threading.Event()
    writer_has_admission = threading.Event()
    writer_has_resource = threading.Event()
    release_writer = threading.Event()
    late_reader_entered = threading.Event()
    errors: list[BaseException] = []

    def first_reader() -> None:
        try:
            with docker_lock.shared_docker_operation_lock(timeout_seconds=5):
                first_reader_entered.set()
                assert release_first_reader.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def writer() -> None:
        admission_fd = os.open(admission_file, os.O_CREAT | os.O_RDWR, 0o600)
        resource_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(admission_fd, fcntl.LOCK_EX)
            writer_has_admission.set()
            fcntl.flock(resource_fd, fcntl.LOCK_EX)
            writer_has_resource.set()
            assert release_writer.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            fcntl.flock(resource_fd, fcntl.LOCK_UN)
            fcntl.flock(admission_fd, fcntl.LOCK_UN)
            os.close(resource_fd)
            os.close(admission_fd)

    def late_reader() -> None:
        try:
            with docker_lock.shared_docker_operation_lock(timeout_seconds=5):
                late_reader_entered.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_thread = threading.Thread(target=first_reader, daemon=True)
    writer_thread = threading.Thread(target=writer, daemon=True)
    late_thread = threading.Thread(target=late_reader, daemon=True)
    first_thread.start()
    assert first_reader_entered.wait(timeout=2)
    writer_thread.start()
    assert writer_has_admission.wait(timeout=2)
    late_thread.start()
    time.sleep(0.1)
    assert late_reader_entered.is_set() is False

    release_first_reader.set()
    assert writer_has_resource.wait(timeout=2)
    time.sleep(0.1)
    assert late_reader_entered.is_set() is False
    release_writer.set()

    for thread in (first_thread, writer_thread, late_thread):
        thread.join(timeout=2)
        assert thread.is_alive() is False
    assert late_reader_entered.is_set() is True
    assert errors == []


@pytest.mark.asyncio
async def test_cancelled_to_thread_keeps_shared_lock_until_worker_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_file = tmp_path / "docker-operation.lock"
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_FILE", str(lock_file))
    monkeypatch.setattr(
        docker_lock.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["docker", "info"], returncode=0
        ),
    )
    entered = threading.Event()
    release = threading.Event()

    def lifecycle() -> None:
        with docker_lock.shared_docker_operation_lock(timeout_seconds=5):
            entered.set()
            assert release.wait(timeout=5)

    task = asyncio.create_task(asyncio.to_thread(lifecycle))
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    exclusive_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    with pytest.raises(BlockingIOError):
        fcntl.flock(exclusive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    release.set()
    deadline = time.monotonic() + 2
    while True:
        try:
            fcntl.flock(exclusive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                pytest.fail(
                    "worker thread did not release its shared lifecycle lock"
                )
            await asyncio.sleep(0.01)
    fcntl.flock(exclusive_fd, fcntl.LOCK_UN)
    os.close(exclusive_fd)
