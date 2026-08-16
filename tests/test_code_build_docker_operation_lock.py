from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

import pytest

from gateway.research_lab import code_build
from research_lab import docker_operation_lock_v2 as docker_lock


ROOT = Path(__file__).resolve().parents[1]
LOCK_HELPER = ROOT / "validator_tee" / "scripts" / "docker_operation_lock_v2.sh"
IMAGE_REF = "example.invalid/private-model@sha256:" + ("a" * 64)


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
        time.sleep(0.01)
""",
    )


def _materialize_parent_app(repo_dir: Path) -> None:
    for relative in code_build._REQUIRED_PARENT_APP_DIRS:
        (repo_dir / relative).mkdir(parents=True, exist_ok=True)
    for relative in code_build._REQUIRED_PARENT_APP_FILES:
        path = repo_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# exact parent\n", encoding="utf-8")


def test_parent_extraction_excludes_attestation_reset_contender(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_file = tmp_path / "docker-operation.lock"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_flock(bin_dir)
    extraction_entered = threading.Event()
    release_extraction = threading.Event()
    extraction_result: dict[str, object] = {}

    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_FILE", str(lock_file))
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS", "5")
    monkeypatch.setattr(
        docker_lock.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["docker", "info"], returncode=0
        ),
    )
    monkeypatch.setattr(
        code_build,
        "_ensure_parent_image_available",
        lambda _image_ref, *, timeout_seconds: None,
    )

    def extract(_image_ref: str, *, repo_dir: Path, timeout_seconds: int) -> None:
        extraction_entered.set()
        assert release_extraction.wait(timeout=5)
        _materialize_parent_app(repo_dir)

    monkeypatch.setattr(code_build, "_extract_parent_image_app", extract)

    def run_extraction() -> None:
        try:
            extraction_result["value"] = code_build._extract_parent_image_source(
                image_digest=IMAGE_REF,
                source_dir=tmp_path / "source",
                timeout_seconds=5,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            extraction_result["error"] = exc

    worker = threading.Thread(target=run_extraction, daemon=True)
    worker.start()
    assert extraction_entered.wait(timeout=2)

    reset_started = tmp_path / "reset.started"
    contender_env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_file),
        "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "5",
    }
    contender = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                f'. "{LOCK_HELPER}"; '
                "leadpoet_acquire_docker_operation_lock_v2; "
                f'touch "{reset_started}"; '
                "leadpoet_release_docker_operation_lock_v2"
            ),
        ],
        env=contender_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.15)
        assert reset_started.exists() is False
        assert contender.poll() is None
        release_extraction.set()
        worker.join(timeout=3)
        assert worker.is_alive() is False
        stdout, stderr = contender.communicate(timeout=3)
        assert contender.returncode == 0, (stdout, stderr)
        assert reset_started.is_file()
        assert "error" not in extraction_result
        assert extraction_result["value"][1]
    finally:
        release_extraction.set()
        worker.join(timeout=1)
        if contender.poll() is None:
            contender.terminate()
            contender.wait(timeout=2)


def test_parent_extraction_lock_timeout_is_retryable_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_file = tmp_path / "docker-operation.lock"
    owner_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_FILE", str(lock_file))
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS", "1")
    try:
        started = time.monotonic()
        with pytest.raises(
            code_build.CodeEditInfraFailureError,
            match="timed out waiting for shared Docker",
        ) as raised:
            code_build._extract_parent_image_source(
                image_digest=IMAGE_REF,
                source_dir=tmp_path / "source",
                timeout_seconds=1,
            )
        assert time.monotonic() - started < 2.5
        assert raised.value.retryable is True
        assert raised.value.failure_stage == "candidate_build_infra_failed"
    finally:
        fcntl.flock(owner_fd, fcntl.LOCK_UN)
        os.close(owner_fd)


def test_private_build_preparation_and_command_share_one_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = False
    calls: list[str] = []

    @contextmanager
    def operation_lock(*, timeout_seconds: int):
        nonlocal held
        assert timeout_seconds == 37
        assert held is False
        held = True
        try:
            yield
        finally:
            held = False

    def prepull(image_ref: str) -> None:
        assert held is True
        assert image_ref == IMAGE_REF
        calls.append("prepull")

    def build(**kwargs: object) -> str:
        assert held is True
        assert kwargs["timeout_seconds"] == 37
        calls.append("build")
        return "complete"

    monkeypatch.setattr(code_build, "_docker_operation_lock_scope", operation_lock)
    monkeypatch.setattr(code_build, "_prepull_parent_image_for_build", prepull)
    monkeypatch.setattr(
        code_build,
        "_run_private_build_cmd_with_infra_retry",
        build,
    )

    assert (
        code_build._run_private_build_under_docker_operation_lock(
            parent_image_ref=IMAGE_REF,
            cmd="docker build && docker push",
            cwd=tmp_path,
            env={},
            timeout_seconds=37,
        )
        == "complete"
    )
    assert calls == ["prepull", "build"]
    assert held is False


def test_timed_out_private_build_reaps_child_before_exclusive_acquires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_file = tmp_path / "docker-operation.lock"
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_FILE", str(lock_file))
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS", "5")
    monkeypatch.setattr(
        docker_lock.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["docker", "info"], returncode=0
        ),
    )
    monkeypatch.setattr(
        code_build,
        "_prepull_parent_image_for_build",
        lambda _image_ref: None,
    )
    pid_file = tmp_path / "child.pid"
    child_script = (
        "from pathlib import Path; import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "sink = os.open(os.devnull, os.O_WRONLY); "
        "os.dup2(sink, 1); os.dup2(sink, 2); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
        "time.sleep(60)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child_script)}"
    build_errors: list[BaseException] = []

    def build() -> None:
        try:
            code_build._run_private_build_under_docker_operation_lock(
                parent_image_ref=IMAGE_REF,
                cmd=command,
                cwd=tmp_path,
                env=os.environ,
                timeout_seconds=1,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            build_errors.append(exc)

    build_thread = threading.Thread(target=build, daemon=True)
    build_thread.start()
    deadline = time.monotonic() + 3
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_file.is_file()
    child_pid = int(pid_file.read_text(encoding="utf-8"))

    admission_file = docker_lock.docker_operation_admission_lock_path(lock_file)
    writer_queued = threading.Event()
    writer_acquired = threading.Event()
    child_alive_at_writer_acquire: list[bool] = []

    def writer() -> None:
        admission_fd = os.open(admission_file, os.O_CREAT | os.O_RDWR, 0o600)
        resource_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(admission_fd, fcntl.LOCK_EX)
            writer_queued.set()
            fcntl.flock(resource_fd, fcntl.LOCK_EX)
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                child_alive_at_writer_acquire.append(False)
            else:
                child_alive_at_writer_acquire.append(True)
            writer_acquired.set()
        finally:
            fcntl.flock(resource_fd, fcntl.LOCK_UN)
            fcntl.flock(admission_fd, fcntl.LOCK_UN)
            os.close(resource_fd)
            os.close(admission_fd)

    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()
    assert writer_queued.wait(timeout=2)
    assert writer_acquired.wait(timeout=5)
    build_thread.join(timeout=2)
    writer_thread.join(timeout=2)

    assert build_thread.is_alive() is False
    assert writer_thread.is_alive() is False
    assert len(build_errors) == 1
    assert isinstance(build_errors[0], code_build.CodeEditImageBuildError)
    assert "timed out" in str(build_errors[0])
    assert child_alive_at_writer_acquire == [False]


def test_docker_operation_lock_is_reentrant_but_rejects_path_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_lock = tmp_path / "first.lock"
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_FILE", str(first_lock))
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(
        docker_lock.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["docker", "info"], returncode=0
        ),
    )

    with code_build._docker_operation_lock_scope(timeout_seconds=1):
        with code_build._docker_operation_lock_scope(timeout_seconds=1):
            assert first_lock.is_file()
        monkeypatch.setenv(
            "LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE",
            str(tmp_path / "other-admission.lock"),
        )
        with pytest.raises(
            code_build.CodeEditInfraFailureError,
            match="nested Docker operation lock path differs",
        ):
            with code_build._docker_operation_lock_scope(timeout_seconds=1):
                pass
        monkeypatch.delenv("LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE")
        monkeypatch.setenv(
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
            str(tmp_path / "second.lock"),
        )
        with pytest.raises(
            code_build.CodeEditInfraFailureError,
            match="nested Docker operation lock path differs",
        ):
            with code_build._docker_operation_lock_scope(timeout_seconds=1):
                pass


@pytest.mark.parametrize(
    ("lock_file", "timeout", "message"),
    (
        ("relative.lock", "1", "must be an absolute path"),
        (None, "0", "timeout must be a positive integer"),
        (None, "not-a-number", "timeout must be a positive integer"),
    ),
)
def test_docker_operation_lock_rejects_ambiguous_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_file: str | None,
    timeout: str,
    message: str,
) -> None:
    monkeypatch.setenv(
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        lock_file or str(tmp_path / "docker-operation.lock"),
    )
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS", timeout)
    with pytest.raises(code_build.CodeEditInfraFailureError, match=message):
        with code_build._docker_operation_lock_scope(timeout_seconds=1):
            pass


def test_docker_readiness_is_polled_under_the_acquired_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        return subprocess.CompletedProcess(
            args=["docker", "info"],
            returncode=1 if attempts < 3 else 0,
            stderr="daemon is restarting" if attempts < 3 else "",
        )

    monkeypatch.setenv(
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE",
        str(tmp_path / "docker-operation.lock"),
    )
    monkeypatch.setattr(docker_lock.subprocess, "run", run)
    monkeypatch.setattr(docker_lock, "DOCKER_DAEMON_READY_POLL_SECONDS", 0.0)
    monkeypatch.setenv("LEADPOET_DOCKER_DAEMON_READY_TIMEOUT_SECONDS", "1")

    with code_build._docker_operation_lock_scope(timeout_seconds=3):
        pass

    assert attempts == 3


def test_two_shared_lifecycles_coexist_and_exclusive_waits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_file = tmp_path / "docker-operation.lock"
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_FILE", str(lock_file))
    monkeypatch.setenv("LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS", "5")
    monkeypatch.setattr(
        docker_lock.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["docker", "info"], returncode=0
        ),
    )
    both_entered = threading.Barrier(3)
    release = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            with docker_lock.shared_docker_operation_lock(timeout_seconds=5):
                both_entered.wait(timeout=2)
                assert release.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(2)]
    for reader_thread in readers:
        reader_thread.start()
    both_entered.wait(timeout=2)

    exclusive_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    with pytest.raises(BlockingIOError):
        fcntl.flock(exclusive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    release.set()
    for reader_thread in readers:
        reader_thread.join(timeout=2)
        assert reader_thread.is_alive() is False
    assert errors == []
    fcntl.flock(exclusive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(exclusive_fd, fcntl.LOCK_UN)
    os.close(exclusive_fd)
