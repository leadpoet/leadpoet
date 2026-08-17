"""Exact N-1 reader versus candidate Docker-maintenance collision rehearsal."""

from __future__ import annotations

import asyncio
from io import BytesIO
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
from unittest.mock import patch


def _wait_for(predicate: Callable[[], bool], *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise RuntimeError("dynamic Docker collision probe exceeded its deadline")
        time.sleep(min(0.02, max(0.001, timeout_seconds / 100.0)))


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o700)


def _install_flock_boundary(directory: Path) -> None:
    """Provide the production flock CLI contract on non-Linux rehearsal hosts."""

    _write_executable(
        directory / "flock",
        f"""#!{sys.executable}
import fcntl
import sys
import time

args = sys.argv[1:]
if len(args) == 2 and args[0] == "-u":
    fcntl.flock(int(args[1]), fcntl.LOCK_UN)
    raise SystemExit(0)
if len(args) == 3 and args[0] == "-w":
    timeout = float(args[1])
    descriptor = int(args[2])
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(0)
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise SystemExit(1)
            time.sleep(min(0.02, max(0.001, timeout / 100.0)))
raise SystemExit(2)
""",
    )


def _json_result(
    completed: subprocess.CompletedProcess[str], *, label: str
) -> dict[str, Any]:
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{label} returned no evidence")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} returned invalid evidence") from exc
    if not isinstance(result, Mapping):
        raise RuntimeError(f"{label} evidence is not an object")
    return dict(result)


def _run_exact_n_minus_one_source_reader(
    *,
    source_root: Path,
    exact_root: Path,
    environment: Mapping[str, str],
    event_path: Path,
    host_detect_path: Path,
    image_digest: str,
    timeout_seconds: int,
    collision_timeout_seconds: float,
) -> subprocess.Popen[str]:
    harness = source_root / "tests/restart_rehearsal/dynamic_rebenchmark_n_minus_one.py"
    child_environment = dict(os.environ)
    child_environment.update(
        {str(key): str(value) for key, value in environment.items()}
    )
    reader_bin = event_path.parent / "n-minus-reader-bin"
    reader_bin.mkdir(mode=0o700)
    _write_executable(
        reader_bin / "docker",
        """#!/bin/sh
if [ "$#" -eq 1 ] && [ "$1" = "info" ]; then
  printf 'info\n' >> "$REHEARSAL_N_MINUS_READER_DOCKER_READY_LOG"
  exit 0
fi
echo "unexpected N-1 Docker readiness operation" >&2
exit 97
""",
    )
    reader_lock_path = event_path.parent / "n-minus-reader.lock"
    reader_ready_log = event_path.parent / "n-minus-reader-docker-ready.log"
    child_environment.update(
        {
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(reader_lock_path),
            "LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE": (
                f"{reader_lock_path}.admission"
            ),
            "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": str(
                max(1, math.ceil(collision_timeout_seconds))
            ),
            "LEADPOET_DOCKER_DAEMON_READY_TIMEOUT_SECONDS": str(
                max(1, math.ceil(collision_timeout_seconds))
            ),
            "REHEARSAL_N_MINUS_READER_DOCKER_READY_LOG": str(reader_ready_log),
            "PATH": str(reader_bin)
            + os.pathsep
            + (child_environment.get("PATH") or os.defpath),
        }
    )
    child_environment["PYTHONPATH"] = str(exact_root)
    child = subprocess.Popen(
        [
            sys.executable,
            str(harness),
            "--docker-source-extraction",
            str(exact_root),
        ],
        cwd=exact_root,
        env=child_environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if child.stdin is None:
        raise RuntimeError("exact N-1 reader stdin is unavailable")
    child.stdin.write(
        json.dumps(
            {
                "event_path": str(event_path),
                "host_detect_path": str(host_detect_path),
                "image_digest": image_digest,
                "timeout_seconds": timeout_seconds,
                "collision_timeout_seconds": collision_timeout_seconds,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    child.stdin.close()
    child.stdin = None
    return child


def _communicate(
    child: subprocess.Popen[str], *, timeout_seconds: float, label: str
) -> subprocess.CompletedProcess[str]:
    try:
        stdout, stderr = child.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        child.kill()
        child.communicate()
        raise RuntimeError(f"{label} exceeded its deadline") from exc
    return subprocess.CompletedProcess(
        args=child.args,
        returncode=int(child.returncode or 0),
        stdout=stdout,
        stderr=stderr,
    )


def _exclusive_probe(
    *,
    shell_lock_path: Path,
    lock_path: Path,
    marker: Path,
    environment: Mapping[str, str],
) -> subprocess.Popen[str]:
    command = """
set -euo pipefail
. "$1"
leadpoet_acquire_docker_operation_lock_v2
touch "$2"
leadpoet_release_docker_operation_lock_v2
"""
    return subprocess.Popen(
        ["bash", "-c", command, "dynamic-exclusive", str(shell_lock_path), str(marker)],
        env={
            **os.environ,
            **{str(key): str(value) for key, value in environment.items()},
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_path),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _held_exclusive_probe(
    *,
    shell_lock_path: Path,
    lock_path: Path,
    acquired_marker: Path,
    release_marker: Path,
    environment: Mapping[str, str],
) -> subprocess.Popen[str]:
    command = """
set -euo pipefail
. "$1"
leadpoet_acquire_docker_operation_lock_v2
touch "$2"
while [ ! -f "$3" ]; do sleep 0.02; done
leadpoet_release_docker_operation_lock_v2
"""
    return subprocess.Popen(
        [
            "bash",
            "-c",
            command,
            "dynamic-exclusive-held",
            str(shell_lock_path),
            str(acquired_marker),
            str(release_marker),
        ],
        env={
            **os.environ,
            **{str(key): str(value) for key, value in environment.items()},
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_path),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _exclusive_lock_is_blocked(lock_path: Path) -> bool:
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _candidate_source_fixture(code_build_module: Any, root: Path) -> tuple[Path, str]:
    from gateway.tee.source_bundle_v2 import compute_private_source_tree_hash

    fixture = root / "candidate-source"
    fixture.mkdir()
    for relative in tuple(code_build_module._REQUIRED_PARENT_APP_DIRS):
        directory = fixture / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rehearsal_source.py").write_text(
            f"SOURCE_PATH = {relative!r}\n", encoding="utf-8"
        )
    for relative in tuple(code_build_module._REQUIRED_PARENT_APP_FILES):
        target = fixture / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"candidate source fixture: {relative}\n", encoding="utf-8")
    return fixture, compute_private_source_tree_hash(fixture)


def _candidate_shared_lifecycle_matrix(
    *, source_root: Path, timeout_seconds: int
) -> dict[str, Any]:
    from gateway.research_lab import code_build as code_build_module
    from gateway.research_lab.code_build import (
        CodeEditBuildError,
        CodeEditInfraFailureError,
    )
    from research_lab import docker_operation_lock_v2 as lock_module

    shell_lock_path = source_root / "validator_tee/scripts/docker_operation_lock_v2.sh"
    matrix_timeout = max(2.0, min(15.0, math.sqrt(max(1, timeout_seconds))))
    with tempfile.TemporaryDirectory(prefix="dynamic-docker-shared-matrix-") as raw:
        root = Path(raw)
        lock_path = root / "docker-operation.lock"
        ready_marker = root / "docker-ready"
        docker_log = root / "docker-info.log"
        fake_docker = root / "docker"
        _install_flock_boundary(root)
        _write_executable(
            fake_docker,
            """#!/bin/sh
printf '%s\\n' "$*" >> "$REHEARSAL_DOCKER_INFO_LOG"
if [ "$1" = "info" ] && [ -f "$REHEARSAL_DOCKER_READY_FILE" ]; then
  exit 0
fi
if [ "$1" = "run" ] && [ "${REHEARSAL_DOCKER_TIMEOUT_MODE:-0}" = "1" ]; then
  touch "$REHEARSAL_DOCKER_TIMEOUT_ENTERED_FILE"
  while :; do sleep 0.02; done
fi
if [ "$1" = "rm" ] && [ "${3:-}" = "dynamic-snapshot-timeout" ]; then
  touch "$(dirname "$0")/named-timeout-cleanup"
  exit 0
fi
if [ "$1" = "run" ] && [ -n "${REHEARSAL_DOCKER_RUN_ENTERED_FILE:-}" ]; then
  touch "$REHEARSAL_DOCKER_RUN_ENTERED_FILE"
  while [ ! -f "$REHEARSAL_DOCKER_RUN_RELEASE_FILE" ]; do sleep 0.02; done
  printf 'strict snapshot lifecycle\\n'
  exit 0
fi
exit 86
""",
        )
        ready_marker.touch()
        fixture, expected_tree_hash = _candidate_source_fixture(code_build_module, root)
        shared_environment = {
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(lock_path),
            "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": str(
                max(1, math.ceil(matrix_timeout))
            ),
            "LEADPOET_DOCKER_DAEMON_READY_TIMEOUT_SECONDS": str(
                max(1, math.ceil(matrix_timeout))
            ),
            "REHEARSAL_DOCKER_READY_FILE": str(ready_marker),
            "REHEARSAL_DOCKER_INFO_LOG": str(docker_log),
            "PATH": str(root) + os.pathsep + os.environ.get("PATH", ""),
        }

        operation_lock = threading.Lock()
        operations: list[tuple[str, tuple[str, ...]]] = []
        both_holders_ready = threading.Event()
        allow_holders_to_finish = threading.Event()
        holder_barrier = threading.Barrier(2, action=both_holders_ready.set)

        def strict_two_holder_run(
            command: list[str], *, cwd: Path, timeout_seconds: int
        ) -> str:
            del cwd, timeout_seconds
            normalized = tuple(str(value) for value in command)
            image = next(
                (
                    value
                    for value in normalized
                    if value.startswith("localhost/dynamic-holder-")
                ),
                "",
            )
            container = next(
                (
                    value.split(":/app/.", 1)[0]
                    for value in normalized
                    if value.startswith("holder-container-")
                ),
                "",
            )
            if image:
                holder = image.rsplit("-", 1)[-1].split("@", 1)[0]
            elif container:
                holder = container.rsplit("-", 1)[-1]
            else:
                raise RuntimeError("candidate holder crossed an unknown boundary")
            with operation_lock:
                operations.append((holder, normalized))
            if normalized[1:3] == ("image", "inspect"):
                holder_barrier.wait(timeout=matrix_timeout)
                if not allow_holders_to_finish.wait(matrix_timeout):
                    raise RuntimeError("candidate shared holder was not released")
                raise CodeEditBuildError("strict cold image cache")
            if normalized[1] == "pull":
                return "strict pull complete"
            if normalized[1] == "create":
                return "holder-container-" + holder
            if normalized[1] == "cp":
                shutil.copytree(fixture, Path(normalized[3]), dirs_exist_ok=True)
                return "strict copy complete"
            if normalized[1:3] == ("rm", "-f"):
                return "strict cleanup complete"
            raise RuntimeError("candidate holder issued an unknown Docker command")

        holder_results: dict[str, tuple[str, list[str]]] = {}
        holder_errors: list[BaseException] = []

        def holder(index: int) -> None:
            label = str(index)
            image = (
                f"localhost/dynamic-holder-{label}@sha256:"
                + hashlib.sha256(label.encode("ascii")).hexdigest()
            )
            try:
                holder_results[label] = code_build_module._extract_parent_image_source(
                    image_digest=image,
                    source_dir=root / f"holder-{label}",
                    timeout_seconds=timeout_seconds,
                )
            except BaseException as exc:
                holder_errors.append(exc)

        with (
            patch.dict(os.environ, shared_environment),
            patch.object(code_build_module, "_run", strict_two_holder_run),
        ):
            holders = [
                threading.Thread(target=holder, args=(index,), daemon=True)
                for index in range(2)
            ]
            for thread in holders:
                thread.start()
            if not both_holders_ready.wait(matrix_timeout):
                raise RuntimeError("candidate shared holders did not overlap")
            exclusive_marker = root / "exclusive-after-two"
            exclusive = _exclusive_probe(
                shell_lock_path=shell_lock_path,
                lock_path=lock_path,
                marker=exclusive_marker,
                environment=shared_environment,
            )
            _wait_for(
                lambda: _exclusive_lock_is_blocked(lock_path),
                timeout_seconds=matrix_timeout,
            )
            if exclusive_marker.exists() or exclusive.poll() is not None:
                raise RuntimeError("exclusive maintenance bypassed shared holders")
            allow_holders_to_finish.set()
            for thread in holders:
                thread.join(matrix_timeout)
                if thread.is_alive():
                    raise RuntimeError("candidate shared holder did not terminate")
            exclusive_result = _communicate(
                exclusive,
                timeout_seconds=matrix_timeout,
                label="exclusive maintenance after two holders",
            )
        if (
            holder_errors
            or set(holder_results) != {"0", "1"}
            or any(
                result[0] != expected_tree_hash for result in holder_results.values()
            )
            or exclusive_result.returncode != 0
            or not exclusive_marker.exists()
        ):
            raise RuntimeError("two candidate shared lifecycles did not settle exactly")
        for label in holder_results:
            observed = [
                tuple(command[1:3])
                for holder_label, command in operations
                if holder_label == label
            ]
            if observed != [
                ("image", "inspect"),
                ("pull", "--platform"),
                ("create", "--platform"),
                ("cp", f"holder-container-{label}:/app/."),
                ("rm", "-f"),
            ]:
                raise RuntimeError("candidate shared lifecycle command order differs")

        from scripts import record_research_lab_dev_snapshots as snapshot_module

        snapshot_entered = root / "snapshot-entered"
        snapshot_release = root / "snapshot-release"
        snapshot_exclusive_marker = root / "exclusive-after-snapshot"
        snapshot_environment = {
            **shared_environment,
            "REHEARSAL_DOCKER_RUN_ENTERED_FILE": str(snapshot_entered),
            "REHEARSAL_DOCKER_RUN_RELEASE_FILE": str(snapshot_release),
        }
        snapshot_result: list[subprocess.CompletedProcess[str]] = []
        snapshot_errors: list[BaseException] = []

        def run_snapshot_lifecycle() -> None:
            try:
                snapshot_result.append(
                    snapshot_module._run_named_docker(
                        [
                            str(fake_docker),
                            "run",
                            "--name",
                            "dynamic-snapshot-lifecycle",
                            "strict-snapshot-image",
                        ],
                        container_name="dynamic-snapshot-lifecycle",
                        input_text="{}",
                        timeout_seconds=max(2, math.ceil(matrix_timeout)),
                        environment=snapshot_environment,
                    )
                )
            except BaseException as exc:
                snapshot_errors.append(exc)

        with patch.dict(os.environ, snapshot_environment):
            snapshot_thread = threading.Thread(
                target=run_snapshot_lifecycle, daemon=True
            )
            snapshot_thread.start()
            _wait_for(snapshot_entered.exists, timeout_seconds=matrix_timeout)
            snapshot_exclusive = _exclusive_probe(
                shell_lock_path=shell_lock_path,
                lock_path=lock_path,
                marker=snapshot_exclusive_marker,
                environment=snapshot_environment,
            )
            _wait_for(
                lambda: _exclusive_lock_is_blocked(lock_path),
                timeout_seconds=matrix_timeout,
            )
            if (
                snapshot_exclusive_marker.exists()
                or snapshot_exclusive.poll() is not None
            ):
                raise RuntimeError(
                    "exclusive maintenance bypassed the snapshot Docker lifecycle"
                )
            snapshot_release.touch()
            snapshot_thread.join(matrix_timeout)
            if snapshot_thread.is_alive():
                raise RuntimeError("snapshot Docker lifecycle did not terminate")
            snapshot_exclusive_result = _communicate(
                snapshot_exclusive,
                timeout_seconds=matrix_timeout,
                label="exclusive maintenance after snapshot lifecycle",
            )
        if (
            snapshot_errors
            or len(snapshot_result) != 1
            or snapshot_result[0].returncode != 0
            or snapshot_exclusive_result.returncode != 0
            or not snapshot_exclusive_marker.exists()
        ):
            raise RuntimeError("snapshot shared Docker lifecycle was not exact")

        named_timeout_entered = root / "named-timeout-entered"
        named_timeout_cleanup = root / "named-timeout-cleanup"
        named_timeout_environment = {
            **shared_environment,
            "REHEARSAL_DOCKER_TIMEOUT_MODE": "1",
            "REHEARSAL_DOCKER_TIMEOUT_ENTERED_FILE": str(named_timeout_entered),
        }
        with patch.dict(os.environ, named_timeout_environment):
            try:
                snapshot_module._run_named_docker(
                    [
                        str(fake_docker),
                        "run",
                        "--name",
                        "dynamic-snapshot-timeout",
                        "strict-snapshot-image",
                    ],
                    container_name="dynamic-snapshot-timeout",
                    input_text="{}",
                    timeout_seconds=1,
                    environment=named_timeout_environment,
                )
            except subprocess.TimeoutExpired:
                named_timeout_rejected = True
            else:
                named_timeout_rejected = False
        named_cleanup_marker = root / "exclusive-after-named-timeout"
        named_cleanup_exclusive = _exclusive_probe(
            shell_lock_path=shell_lock_path,
            lock_path=lock_path,
            marker=named_cleanup_marker,
            environment=shared_environment,
        )
        named_cleanup_exclusive_result = _communicate(
            named_cleanup_exclusive,
            timeout_seconds=matrix_timeout,
            label="exclusive maintenance after named timeout cleanup",
        )
        if (
            not named_timeout_rejected
            or not named_timeout_entered.exists()
            or not named_timeout_cleanup.exists()
            or named_cleanup_exclusive_result.returncode != 0
            or not named_cleanup_marker.exists()
        ):
            raise RuntimeError("named Docker timeout cleanup was not serialized")

        timeout_acquired = root / "timeout-exclusive-acquired"
        timeout_release = root / "timeout-exclusive-release"
        held_exclusive = _held_exclusive_probe(
            shell_lock_path=shell_lock_path,
            lock_path=lock_path,
            acquired_marker=timeout_acquired,
            release_marker=timeout_release,
            environment=shared_environment,
        )
        _wait_for(timeout_acquired.exists, timeout_seconds=matrix_timeout)
        timeout_boundary_calls: list[list[str]] = []

        def unexpected_timeout_run(command: list[str], **_kwargs: Any) -> str:
            timeout_boundary_calls.append([str(value) for value in command])
            raise RuntimeError("timed-out holder reached Docker")

        timeout_environment = {
            **shared_environment,
            "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": "1",
        }
        try:
            with (
                patch.dict(os.environ, timeout_environment),
                patch.object(code_build_module, "_run", unexpected_timeout_run),
            ):
                try:
                    code_build_module._extract_parent_image_source(
                        image_digest="localhost/dynamic-timeout@sha256:" + "1" * 64,
                        source_dir=root / "timeout-source",
                        timeout_seconds=timeout_seconds,
                    )
                except CodeEditInfraFailureError as exc:
                    timeout_rejected = "timed out waiting" in str(exc)
                else:
                    timeout_rejected = False
        finally:
            timeout_release.touch()
            held_result = _communicate(
                held_exclusive,
                timeout_seconds=matrix_timeout,
                label="held exclusive timeout probe",
            )
        if (
            not timeout_rejected
            or timeout_boundary_calls
            or held_result.returncode != 0
        ):
            raise RuntimeError("candidate shared-lock timeout was not fail-closed")

        ready_marker.unlink()
        daemon_boundary_calls: list[list[str]] = []

        def unexpected_daemon_run(command: list[str], **_kwargs: Any) -> str:
            daemon_boundary_calls.append([str(value) for value in command])
            raise RuntimeError("daemon-not-ready holder reached Docker")

        daemon_environment = {
            **shared_environment,
            "LEADPOET_DOCKER_DAEMON_READY_TIMEOUT_SECONDS": "1",
        }
        with (
            patch.dict(os.environ, daemon_environment),
            patch.object(code_build_module, "_run", unexpected_daemon_run),
        ):
            try:
                code_build_module._extract_parent_image_source(
                    image_digest="localhost/dynamic-daemon@sha256:" + "2" * 64,
                    source_dir=root / "daemon-source",
                    timeout_seconds=timeout_seconds,
                )
            except CodeEditInfraFailureError as exc:
                daemon_not_ready_rejected = "did not become ready" in str(exc)
            else:
                daemon_not_ready_rejected = False
        ready_marker.touch()
        daemon_release_marker = root / "exclusive-after-daemon"
        daemon_release = _exclusive_probe(
            shell_lock_path=shell_lock_path,
            lock_path=lock_path,
            marker=daemon_release_marker,
            environment=shared_environment,
        )
        daemon_release_result = _communicate(
            daemon_release,
            timeout_seconds=matrix_timeout,
            label="exclusive maintenance after daemon failure",
        )
        if (
            not daemon_not_ready_rejected
            or daemon_boundary_calls
            or daemon_release_result.returncode != 0
            or not daemon_release_marker.exists()
        ):
            raise RuntimeError("daemon-not-ready failure retained the shared lock")

        cancellation_entered = threading.Event()
        cancellation_release = threading.Event()
        cancellation_complete = threading.Event()

        def cancellation_run(
            command: list[str], *, cwd: Path, timeout_seconds: int
        ) -> str:
            del cwd, timeout_seconds
            normalized = [str(value) for value in command]
            if normalized[1:3] == ["image", "inspect"]:
                cancellation_entered.set()
                if not cancellation_release.wait(matrix_timeout):
                    raise RuntimeError("cancelled Docker holder was not released")
                raise CodeEditBuildError("strict cold image cache")
            if normalized[1] == "pull":
                return "strict pull complete"
            if normalized[1] == "create":
                return "cancelled-holder-container"
            if normalized[1] == "cp":
                shutil.copytree(fixture, Path(normalized[3]), dirs_exist_ok=True)
                return "strict copy complete"
            if normalized[1:3] == ["rm", "-f"]:
                cancellation_complete.set()
                return "strict cleanup complete"
            raise RuntimeError("cancelled holder issued an unknown Docker command")

        async def cancellation_case() -> bool:
            def run_holder() -> tuple[str, list[str]]:
                try:
                    return code_build_module._extract_parent_image_source(
                        image_digest="localhost/dynamic-cancel@sha256:" + "3" * 64,
                        source_dir=root / "cancel-source",
                        timeout_seconds=timeout_seconds,
                    )
                finally:
                    cancellation_complete.set()

            task = asyncio.create_task(asyncio.to_thread(run_holder))
            entered = await asyncio.to_thread(cancellation_entered.wait, matrix_timeout)
            if not entered:
                raise RuntimeError("cancelled holder never acquired shared access")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                cancellation_observed = True
            else:
                cancellation_observed = False
            marker = root / "exclusive-after-cancel"
            exclusive = _exclusive_probe(
                shell_lock_path=shell_lock_path,
                lock_path=lock_path,
                marker=marker,
                environment=shared_environment,
            )
            await asyncio.to_thread(
                _wait_for,
                lambda: _exclusive_lock_is_blocked(lock_path),
                timeout_seconds=matrix_timeout,
            )
            if marker.exists() or exclusive.poll() is not None:
                raise RuntimeError("task cancellation released a live Docker holder")
            cancellation_release.set()
            completed = await asyncio.to_thread(
                cancellation_complete.wait, matrix_timeout
            )
            if not completed:
                raise RuntimeError("cancelled holder did not finish cleanup")
            result = await asyncio.to_thread(
                _communicate,
                exclusive,
                timeout_seconds=matrix_timeout,
                label="exclusive maintenance after cancellation",
            )
            return bool(
                cancellation_observed and result.returncode == 0 and marker.exists()
            )

        with (
            patch.dict(os.environ, shared_environment),
            patch.object(code_build_module, "_run", cancellation_run),
        ):
            cancellation_safe = asyncio.run(cancellation_case())
        if not cancellation_safe:
            raise RuntimeError("cancelled shared lifecycle released early")

        cleanup_failure_seen = False

        def cleanup_failure_run(
            command: list[str], *, cwd: Path, timeout_seconds: int
        ) -> str:
            del cwd, timeout_seconds
            normalized = [str(value) for value in command]
            if normalized[1:3] == ["image", "inspect"]:
                raise CodeEditBuildError("strict cold image cache")
            if normalized[1] == "pull":
                return "strict pull complete"
            if normalized[1] == "create":
                return "cleanup-failure-container"
            if normalized[1] == "cp":
                shutil.copytree(fixture, Path(normalized[3]), dirs_exist_ok=True)
                return "strict copy complete"
            if normalized[1:3] == ["rm", "-f"]:
                nonlocal cleanup_failure_seen
                cleanup_failure_seen = True
                raise CodeEditBuildError("strict cleanup failure")
            raise RuntimeError("cleanup-failure holder issued an unknown command")

        with (
            patch.dict(os.environ, shared_environment),
            patch.object(code_build_module, "_run", cleanup_failure_run),
        ):
            cleanup_result = code_build_module._extract_parent_image_source(
                image_digest="localhost/dynamic-cleanup@sha256:" + "4" * 64,
                source_dir=root / "cleanup-source",
                timeout_seconds=timeout_seconds,
            )
        cleanup_marker = root / "exclusive-after-cleanup"
        cleanup_exclusive = _exclusive_probe(
            shell_lock_path=shell_lock_path,
            lock_path=lock_path,
            marker=cleanup_marker,
            environment=shared_environment,
        )
        cleanup_exclusive_result = _communicate(
            cleanup_exclusive,
            timeout_seconds=matrix_timeout,
            label="exclusive maintenance after cleanup failure",
        )
        if (
            not cleanup_failure_seen
            or cleanup_result[0] != expected_tree_hash
            or cleanup_exclusive_result.returncode != 0
            or not cleanup_marker.exists()
        ):
            raise RuntimeError("cleanup failure retained shared Docker access")

        source_paths = tuple(lock_module.shared_docker_operation_source_paths())
        if not source_paths or any(
            not (source_root / path).is_file() for path in source_paths
        ):
            raise RuntimeError("candidate shared Docker source inventory is incomplete")
        return {
            "shared_holder_count": len(holder_results),
            "shared_lifecycle_operation_count": len(operations),
            "exclusive_waited_for_both_shared_holders": True,
            "shared_lock_timeout_fail_closed": timeout_rejected,
            "daemon_not_ready_fail_closed": daemon_not_ready_rejected,
            "cancellation_preserved_live_holder": cancellation_safe,
            "cleanup_failure_released_shared_holder": cleanup_failure_seen,
            "snapshot_shared_lifecycle_excluded_maintenance": True,
            "named_container_timeout_cleanup_before_unlock": True,
            "shared_lock_source_paths": list(source_paths),
            "exact_candidate_code_build_module": str(
                Path(code_build_module.__file__).resolve().relative_to(source_root)
            ),
            "exact_candidate_lock_module": str(
                Path(lock_module.__file__).resolve().relative_to(source_root)
            ),
            "exact_candidate_snapshot_module": str(
                Path(snapshot_module.__file__).resolve().relative_to(source_root)
            ),
        }


def exercise_dynamic_docker_collision(
    *,
    source_root: Path,
    exact_root: Path,
    from_sha: str,
    candidate_sha: str,
    launch_environment: Mapping[str, str],
    scoring_worker_count: int,
    scoring_memory_floor_mib: int,
    model_timeout_seconds: int,
) -> dict[str, Any]:
    """Run the exact collision and candidate shared-lifecycle failure matrix."""

    normalized_root = source_root.resolve()
    gateway_restart_source = (normalized_root / "gw_restart.sh").read_text(
        encoding="utf-8"
    )
    reset_start = gateway_restart_source.index(
        "reset_orphaned_docker_storage_if_needed()"
    )
    reset_end = gateway_restart_source.index("\nensure_docker_ready()", reset_start)
    reset_source = gateway_restart_source[reset_start:reset_end]
    emergency_start = gateway_restart_source.index("emergency_disk_preflight()")
    emergency_end = gateway_restart_source.index(
        "\nstop_research_lab_private_model_containers()", emergency_start
    )
    emergency_source = gateway_restart_source[emergency_start:emergency_end]
    guarded_reclaim_path = "validator_tee/scripts/reclaim_docker_storage_v2.sh"
    if (
        guarded_reclaim_path not in reset_source
        or guarded_reclaim_path not in emergency_source
        or "systemctl stop docker" in reset_source
        or "rm -rf /var/lib/docker" in reset_source
        or "docker system prune" in emergency_source
    ):
        raise RuntimeError(
            "gateway emergency reset does not delegate to guarded reclaim"
        )
    collision_timeout = max(
        5.0,
        min(30.0, math.sqrt(max(1, model_timeout_seconds)) * 3.0),
    )
    live_floor_bytes = max(1, int(scoring_memory_floor_mib)) * 1024 * 1024
    available_bytes = live_floor_bytes + max(
        1, live_floor_bytes // max(1, int(scoring_worker_count))
    )
    image_digest = (
        "localhost/dynamic-n-minus-one@sha256:"
        + hashlib.sha256(from_sha.encode("ascii")).hexdigest()
    )

    with tempfile.TemporaryDirectory(prefix="dynamic-docker-collision-") as raw:
        root = Path(raw)
        fake_bin = root / "bin"
        fake_bin.mkdir()
        _install_flock_boundary(fake_bin)
        proc_root = root / "proc"
        gateway_pid = max(1, os.getpid() + max(1, int(scoring_worker_count)))
        gateway_proc = proc_root / str(gateway_pid)
        gateway_proc.mkdir(parents=True)
        (gateway_proc / "status").write_text(
            "Name:\tpython\nPPid:\t1\n", encoding="utf-8"
        )
        (gateway_proc / "cmdline").write_bytes(
            os.fsencode(sys.executable) + b"\0-u\0-m\0gateway.main\0"
        )

        event_path = root / "events.log"
        guard_ready = root / "guard-ready"
        reader_started = root / "reader-started"
        host_detect = root / "host-detect"
        docker_log = root / "candidate-docker.log"
        sudo_log = root / "candidate-sudo.log"
        wrapper = fake_bin / "python3"
        _write_executable(
            wrapper,
            f"""#!{sys.executable}
import os
from pathlib import Path
import subprocess
import sys
import time

args = sys.argv[1:]
real = os.environ["REHEARSAL_REAL_PYTHON"]
if "--wait" in args and "validator_tee.host.docker_operation_guard_v2" in args:
    completed = subprocess.run([real, *args], text=True, capture_output=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    with open(os.environ["REHEARSAL_COLLISION_EVENT_PATH"], "a", encoding="utf-8") as handle:
        handle.write("cleanup_guard_ready\\n")
        handle.flush()
    Path(os.environ["REHEARSAL_GUARD_READY_PATH"]).touch()
    deadline = time.monotonic() + float(os.environ["REHEARSAL_COLLISION_WAIT_SECONDS"])
    reader = Path(os.environ["REHEARSAL_READER_STARTED_PATH"])
    while not reader.exists():
        if time.monotonic() >= deadline:
            raise SystemExit("late N-1 reader did not start")
        time.sleep(0.01)
elif "--detect-exact-host-gateway" in args:
    with open(os.environ["REHEARSAL_COLLISION_EVENT_PATH"], "a", encoding="utf-8") as handle:
        handle.write("host_gateway_detect\\n")
        handle.flush()
    Path(os.environ["REHEARSAL_HOST_DETECT_PATH"]).touch()
os.execv(real, [real, *args])
""",
        )
        _write_executable(
            fake_bin / "df",
            """#!/bin/sh
printf 'Avail\\n%s\\n' "$REHEARSAL_AVAILABLE_BYTES"
""",
        )
        _write_executable(
            fake_bin / "docker",
            """#!/bin/sh
printf '%s\\n' "$*" >> "$REHEARSAL_CANDIDATE_DOCKER_LOG"
exit 97
""",
        )
        _write_executable(
            fake_bin / "sudo",
            """#!/bin/sh
printf '%s\\n' "$*" >> "$REHEARSAL_CANDIDATE_SUDO_LOG"
exit 98
""",
        )

        reclaim_environment = {
            **os.environ,
            **{str(key): str(value) for key, value in launch_environment.items()},
            "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
            "PYTHONPATH": str(normalized_root),
            "LEADPOET_PROC_ROOT": str(proc_root),
            "LEADPOET_DOCKER_ALLOW_NONSTANDARD_PROC_ROOT": "1",
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(root / "release.lock"),
            "LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS": str(
                max(1, math.ceil(collision_timeout))
            ),
            "VALIDATOR_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES": str(live_floor_bytes),
            "VALIDATOR_DOCKER_MIN_FREE_BYTES": str(available_bytes + 1),
            "REHEARSAL_AVAILABLE_BYTES": str(available_bytes),
            "REHEARSAL_REAL_PYTHON": sys.executable,
            "REHEARSAL_COLLISION_EVENT_PATH": str(event_path),
            "REHEARSAL_GUARD_READY_PATH": str(guard_ready),
            "REHEARSAL_READER_STARTED_PATH": str(reader_started),
            "REHEARSAL_HOST_DETECT_PATH": str(host_detect),
            "REHEARSAL_COLLISION_WAIT_SECONDS": str(collision_timeout),
            "REHEARSAL_CANDIDATE_DOCKER_LOG": str(docker_log),
            "REHEARSAL_CANDIDATE_SUDO_LOG": str(sudo_log),
        }
        reclaim = subprocess.Popen(
            [
                "bash",
                str(
                    normalized_root
                    / "validator_tee/scripts/reclaim_docker_storage_v2.sh"
                ),
            ],
            cwd=normalized_root,
            env=reclaim_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def guard_ready_or_fail() -> bool:
            if guard_ready.exists():
                return True
            if reclaim.poll() is None:
                return False
            failed = _communicate(
                reclaim,
                timeout_seconds=collision_timeout,
                label="candidate Docker reclaim pre-guard",
            )
            raise RuntimeError(
                "candidate Docker reclaim exited before its guard: "
                + str(failed.stderr or failed.stdout)[-500:]
            )

        _wait_for(guard_ready_or_fail, timeout_seconds=collision_timeout)
        reader = _run_exact_n_minus_one_source_reader(
            source_root=normalized_root,
            exact_root=exact_root,
            environment=launch_environment,
            event_path=event_path,
            host_detect_path=host_detect,
            image_digest=image_digest,
            timeout_seconds=model_timeout_seconds,
            collision_timeout_seconds=collision_timeout,
        )

        def reader_started_or_fail() -> bool:
            if "n_minus_reader_started" in (
                event_path.read_text(encoding="utf-8")
                if event_path.exists()
                else ""
            ):
                return True
            if reader.poll() is None:
                return False
            failed = _communicate(
                reader,
                timeout_seconds=collision_timeout,
                label="exact N-1 Docker source reader pre-start",
            )
            reader_started.touch()
            reclaim_cleanup_detail = ""
            try:
                reclaimed = _communicate(
                    reclaim,
                    timeout_seconds=collision_timeout,
                    label="candidate Docker reclaim after reader failure",
                )
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve root failure
                reclaim_cleanup_detail = (
                    "; reclaim cleanup failed: " + type(cleanup_exc).__name__
                )
            else:
                if reclaimed.returncode:
                    reclaim_cleanup_detail = (
                        "; reclaim cleanup exit=" + str(reclaimed.returncode)
                    )
            raise RuntimeError(
                "exact N-1 Docker source reader exited before its first "
                "Docker operation: "
                + str(failed.stderr or failed.stdout)[-1000:]
                + reclaim_cleanup_detail
            )

        _wait_for(reader_started_or_fail, timeout_seconds=collision_timeout)
        reader_started.touch()
        reader_result = _json_result(
            _communicate(
                reader,
                timeout_seconds=collision_timeout,
                label="exact N-1 Docker source reader",
            ),
            label="exact N-1 Docker source reader",
        )
        reader_ready_calls = (
            (root / "n-minus-reader-docker-ready.log")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        if reader_ready_calls != ["info"]:
            raise RuntimeError("N-1 Docker readiness boundary differs")
        reclaim_result = _communicate(
            reclaim,
            timeout_seconds=collision_timeout,
            label="candidate Docker reclaim collision",
        )
        events = event_path.read_text(encoding="utf-8").splitlines()
        required_events = (
            "cleanup_guard_ready",
            "n_minus_reader_started",
            "host_gateway_detect",
            "n_minus_source_extracted",
        )
        if any(event not in events for event in required_events):
            raise RuntimeError("Docker collision event ledger is incomplete")
        if not (
            events.index("cleanup_guard_ready")
            < events.index("n_minus_reader_started")
            < events.index("host_gateway_detect")
            < events.index("n_minus_source_extracted")
        ):
            raise RuntimeError("Docker cleanup guard/late-reader ordering differs")
        if (
            reclaim_result.returncode != 0
            or "storage maintenance deferred" not in reclaim_result.stdout
            or f'"pid": {gateway_pid}' not in reclaim_result.stdout
            or docker_log.exists()
            or sudo_log.exists()
            or reader_result.get("strict_docker_boundary_executed") is not True
            or reader_result.get("exact_model_authority_module")
            != "gateway/research_lab/model_authority_v2.py"
            or reader_result.get("exact_code_build_module")
            != "gateway/research_lab/code_build.py"
        ):
            raise RuntimeError("host-live collision did not preserve the N-1 reader")

    shared_matrix = _candidate_shared_lifecycle_matrix(
        source_root=normalized_root,
        timeout_seconds=model_timeout_seconds,
    )
    return {
        "from_sha": from_sha,
        "candidate_sha": candidate_sha,
        "host_gateway_pid": gateway_pid,
        "scoring_worker_count": int(scoring_worker_count),
        "scoring_memory_floor_mib": int(scoring_memory_floor_mib),
        "live_runtime_floor_bytes": live_floor_bytes,
        "available_runtime_bytes": available_bytes,
        "collision_events": events,
        "n_minus_one_source_extraction": reader_result,
        **shared_matrix,
        "cleanup_guard_then_late_reader_order_exact": True,
        "host_live_prevented_docker_stop_or_reset": True,
        "candidate_gateway_emergency_uses_guarded_reclaim": True,
        "first_activation_requires_preexisting_disk_reserve": True,
        "dynamic_docker_collision_exact": True,
    }


def exercise_dynamic_docker_collision_from_releases(
    *,
    source_root: Path,
    from_sha: str,
    candidate_sha: str,
    launch_environment: Mapping[str, str],
    scoring_worker_count: int,
    scoring_memory_floor_mib: int,
    model_timeout_seconds: int,
) -> dict[str, Any]:
    """Extract exact N-1 and run the standalone behavior-contract scenario."""

    from dynamic_rebenchmark_workflow import (
        git_blob_identity,
        transition_source_paths_by_commit,
    )

    normalized_root = source_root.resolve()
    with tempfile.TemporaryDirectory(prefix="dynamic-docker-exact-release-") as raw:
        exact_root = Path(raw)
        archive = subprocess.run(
            ["git", "archive", "--format=tar", from_sha],
            cwd=normalized_root,
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(exact_root)
        result = exercise_dynamic_docker_collision(
            source_root=normalized_root,
            exact_root=exact_root,
            from_sha=from_sha,
            candidate_sha=candidate_sha,
            launch_environment=launch_environment,
            scoring_worker_count=scoring_worker_count,
            scoring_memory_floor_mib=scoring_memory_floor_mib,
            model_timeout_seconds=model_timeout_seconds,
        )
    inventory = transition_source_paths_by_commit(
        from_sha=from_sha,
        candidate_sha=candidate_sha,
    )
    result["source_inventory"] = {
        commit: list(paths) for commit, paths in inventory.items()
    }
    source_identities = [
        git_blob_identity(normalized_root, commit, path)
        for commit, paths in inventory.items()
        for path in paths
    ]
    if any(
        identity["sha256"]
        != hashlib.sha256((normalized_root / identity["path"]).read_bytes()).hexdigest()
        for identity in source_identities
        if identity["commit_sha"] == candidate_sha
    ):
        raise RuntimeError(
            "candidate Docker lifecycle source differs from its exact commit"
        )
    result["source_identities"] = source_identities
    return result


__all__ = [
    "exercise_dynamic_docker_collision",
    "exercise_dynamic_docker_collision_from_releases",
]
