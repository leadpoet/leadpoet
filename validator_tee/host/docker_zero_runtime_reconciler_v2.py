"""Refresh dockerd metadata after raw cleanup under an exact empty-runtime guard."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Callable, Optional, Sequence


class DockerZeroRuntimeReconcilerV2Error(RuntimeError):
    """Raised when dockerd metadata cannot be reconciled without data loss."""


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
Sleeper = Callable[[float], None]
LockVerifier = Callable[..., None]
Clock = Callable[[], float]

_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_INVOCATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class _ServiceIdentity:
    main_pid: int
    invocation_id: str


@dataclass(frozen=True)
class _RootIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class _EmptyRuntimeSnapshot:
    containerd_service: _ServiceIdentity
    docker_root: str
    docker_root_identity: _RootIdentity
    docker_service: _ServiceIdentity
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class DockerZeroRuntimeReconcileResult:
    docker_root: str
    image_count: int
    image_manifest_hash: str
    root_device: int
    root_inode: int

    def as_dict(self) -> dict[str, object]:
        return {
            "container_count": 0,
            "containerd_container_count": 0,
            "containerd_task_count": 0,
            "docker_root": self.docker_root,
            "image_count": self.image_count,
            "image_manifest_hash": self.image_manifest_hash,
            "moby_shim_count": 0,
            "restart_performed": True,
            "root_device": self.root_device,
            "root_inode": self.root_inode,
            "schema_version": "leadpoet.docker_zero_runtime_reconcile.v1",
            "status": "ready",
        }


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _checked_output(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    label: str,
) -> str:
    try:
        result = runner(command)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DockerZeroRuntimeReconcilerV2Error(f"{label} failed") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise DockerZeroRuntimeReconcilerV2Error(
            f"{label} failed" + (f": {detail}" if detail else "")
        )
    return result.stdout


def _deadline_runner(
    runner: CommandRunner,
    *,
    deadline: float,
    clock: Clock,
) -> CommandRunner:
    def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if clock() >= deadline:
            raise DockerZeroRuntimeReconcilerV2Error(
                "Docker reconciliation deadline was exhausted"
            )
        result = runner(command)
        if clock() >= deadline:
            raise DockerZeroRuntimeReconcilerV2Error(
                "Docker reconciliation command exceeded its deadline"
            )
        return result

    return run


def _service_identity(
    runner: CommandRunner,
    service: str,
) -> _ServiceIdentity:
    raw = _checked_output(
        runner,
        [
            "systemctl",
            "show",
            service,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=InvocationID",
            "--no-pager",
        ],
        label=f"{service} identity",
    )
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            raise DockerZeroRuntimeReconcilerV2Error(
                f"{service} identity is malformed"
            )
        key, value = line.split("=", 1)
        if key in fields:
            raise DockerZeroRuntimeReconcilerV2Error(
                f"{service} identity contains duplicate fields"
            )
        fields[key] = value
    if set(fields) != {
        "LoadState",
        "ActiveState",
        "SubState",
        "MainPID",
        "InvocationID",
    }:
        raise DockerZeroRuntimeReconcilerV2Error(
            f"{service} identity is incomplete"
        )
    if (
        fields["LoadState"] != "loaded"
        or fields["ActiveState"] != "active"
        or fields["SubState"] != "running"
    ):
        raise DockerZeroRuntimeReconcilerV2Error(
            f"{service} is not loaded, active, and running"
        )
    if re.fullmatch(r"[1-9][0-9]*", fields["MainPID"]) is None:
        raise DockerZeroRuntimeReconcilerV2Error(
            f"{service} main pid is malformed"
        )
    invocation_id = fields["InvocationID"].lower()
    if _INVOCATION_ID_RE.fullmatch(invocation_id) is None:
        raise DockerZeroRuntimeReconcilerV2Error(
            f"{service} invocation identity is malformed"
        )
    return _ServiceIdentity(
        main_pid=int(fields["MainPID"]),
        invocation_id=invocation_id,
    )


def _validated_ids(raw: str, *, image: bool, field: str) -> tuple[str, ...]:
    rows = tuple(value.strip() for value in raw.splitlines() if value.strip())
    values = tuple(sorted(set(rows))) if image else tuple(sorted(rows))
    if not image and len(values) != len(set(values)):
        raise DockerZeroRuntimeReconcilerV2Error(f"{field} contains duplicates")
    pattern = _IMAGE_ID_RE if image else _CONTAINER_ID_RE
    if any(pattern.fullmatch(value) is None for value in values):
        raise DockerZeroRuntimeReconcilerV2Error(
            f"{field} contains a malformed identity"
        )
    return values


def _require_empty_ids(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    label: str,
) -> None:
    values = _validated_ids(
        _checked_output(runner, command, label=label),
        image=False,
        field=label,
    )
    if values:
        raise DockerZeroRuntimeReconcilerV2Error(
            f"refusing dockerd reconciliation while {label} is nonempty"
        )


def _moby_shim_count(runner: CommandRunner) -> int:
    command = [
        "pgrep",
        "-fc",
        "^/usr/bin/containerd-shim-runc-v2 -namespace moby ",
    ]
    try:
        result = runner(command)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DockerZeroRuntimeReconcilerV2Error(
            "moby shim inventory failed"
        ) from exc
    if result.returncode not in {0, 1}:
        detail = (result.stderr or result.stdout or "").strip()
        raise DockerZeroRuntimeReconcilerV2Error(
            "moby shim inventory failed" + (f": {detail}" if detail else "")
        )
    raw = result.stdout
    if re.fullmatch(r"(?:0|[1-9][0-9]*)\n?", raw) is None:
        raise DockerZeroRuntimeReconcilerV2Error(
            "moby shim inventory is malformed"
        )
    count = int(raw.removesuffix("\n"))
    if result.returncode == 1 and count != 0:
        raise DockerZeroRuntimeReconcilerV2Error(
            "moby shim inventory status is inconsistent"
        )
    return count


def _root_identity(expected_root: Path) -> _RootIdentity:
    if not expected_root.is_absolute() or expected_root.is_symlink():
        raise DockerZeroRuntimeReconcilerV2Error(
            "Docker data-root path is unsafe"
        )
    try:
        metadata = expected_root.lstat()
    except OSError as exc:
        raise DockerZeroRuntimeReconcilerV2Error(
            "Docker data-root is unreadable"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise DockerZeroRuntimeReconcilerV2Error(
            "Docker data-root is not a directory"
        )
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise DockerZeroRuntimeReconcilerV2Error(
            "Docker data-root ownership or mode is unsafe"
        )
    if metadata.st_dev <= 0 or metadata.st_ino <= 0:
        raise DockerZeroRuntimeReconcilerV2Error(
            "Docker data-root identity is malformed"
        )
    return _RootIdentity(device=metadata.st_dev, inode=metadata.st_ino)


def _empty_runtime_snapshot(
    runner: CommandRunner,
    *,
    expected_root: Path,
) -> _EmptyRuntimeSnapshot:
    docker_service = _service_identity(runner, "docker.service")
    containerd_service = _service_identity(runner, "containerd.service")
    observed_root = _checked_output(
        runner,
        ["docker", "info", "--format", "{{.DockerRootDir}}"],
        label="Docker data-root inventory",
    ).strip()
    if observed_root != str(expected_root):
        raise DockerZeroRuntimeReconcilerV2Error(
            f"refusing unexpected Docker data-root: {observed_root}"
        )
    root_identity = _root_identity(expected_root)
    _require_empty_ids(
        runner,
        ["docker", "ps", "-aq", "--no-trunc"],
        label="Docker container inventory",
    )
    _require_empty_ids(
        runner,
        ["ctr", "-n", "moby", "containers", "list", "-q"],
        label="containerd container inventory",
    )
    _require_empty_ids(
        runner,
        ["ctr", "-n", "moby", "tasks", "list", "-q"],
        label="containerd task inventory",
    )
    shim_count = _moby_shim_count(runner)
    if shim_count != 0:
        raise DockerZeroRuntimeReconcilerV2Error(
            "refusing dockerd reconciliation while moby shims are present"
        )
    image_ids = _validated_ids(
        _checked_output(
            runner,
            ["docker", "image", "ls", "-aq", "--no-trunc"],
            label="Docker image inventory",
        ),
        image=True,
        field="Docker image inventory",
    )
    return _EmptyRuntimeSnapshot(
        containerd_service=containerd_service,
        docker_root=observed_root,
        docker_root_identity=root_identity,
        docker_service=docker_service,
        image_ids=image_ids,
    )


def _canonical_lock_path(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise DockerZeroRuntimeReconcilerV2Error(f"{label} path is invalid")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DockerZeroRuntimeReconcilerV2Error(
            f"{label} path is invalid"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise DockerZeroRuntimeReconcilerV2Error(f"{label} path is invalid")
    return Path(os.path.realpath(path))


def _require_exclusively_locked(path: Path, *, label: str) -> None:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DockerZeroRuntimeReconcilerV2Error(
            f"{label} cannot be inspected"
        ) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        raise DockerZeroRuntimeReconcilerV2Error(f"{label} is not held")
    finally:
        os.close(descriptor)


def verify_docker_operation_locks_v2(
    *,
    lock_file: Path,
    admission_lock_file: Path,
    owner_pid: int,
    proc_root: Path = Path("/proc"),
) -> None:
    """Require the exact parent shell to own both exclusive Docker locks."""

    lock_path = _canonical_lock_path(lock_file, label="Docker operation lock")
    admission_path = _canonical_lock_path(
        admission_lock_file,
        label="Docker operation admission lock",
    )
    if lock_path == admission_path:
        raise DockerZeroRuntimeReconcilerV2Error(
            "Docker operation lock paths must differ"
        )
    if owner_pid < 1:
        raise DockerZeroRuntimeReconcilerV2Error(
            "Docker operation lock owner is invalid"
        )
    for descriptor, expected, label in (
        (7, lock_path, "Docker operation lock"),
        (8, admission_path, "Docker operation admission lock"),
    ):
        try:
            raw_target = os.readlink(proc_root / str(owner_pid) / "fd" / str(descriptor))
        except OSError as exc:
            raise DockerZeroRuntimeReconcilerV2Error(
                f"{label} owner descriptor is unreadable"
            ) from exc
        if raw_target.endswith(" (deleted)"):
            raise DockerZeroRuntimeReconcilerV2Error(
                f"{label} owner descriptor is stale"
            )
        observed = Path(os.path.realpath(raw_target))
        if observed != expected:
            raise DockerZeroRuntimeReconcilerV2Error(
                f"{label} is not owned by the declared parent"
            )
        _require_exclusively_locked(expected, label=label)


def _recover_docker_service(
    runner: CommandRunner,
    *,
    expected_containerd: _ServiceIdentity,
    expected_root: Path,
    ready_attempts: int,
    sleeper: Sleeper,
    deadline: float,
    clock: Clock,
) -> Optional[str]:
    errors: list[str] = []
    try:
        result = runner(["systemctl", "start", "docker.service"])
    except (
        DockerZeroRuntimeReconcilerV2Error,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        errors.append(f"Docker recovery start failed: {type(exc).__name__}")
    else:
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            errors.append(
                "Docker recovery start failed" + (f": {detail}" if detail else "")
            )
    for attempt in range(1, ready_attempts + 1):
        if clock() >= deadline:
            errors.append("Docker recovery deadline was exhausted")
            break
        try:
            if _service_identity(runner, "containerd.service") != expected_containerd:
                raise DockerZeroRuntimeReconcilerV2Error(
                    "containerd identity changed during Docker recovery"
                )
            _service_identity(runner, "docker.service")
            observed_root = _checked_output(
                runner,
                ["docker", "info", "--format", "{{.DockerRootDir}}"],
                label="Docker recovery data-root inventory",
            ).strip()
            if observed_root != str(expected_root):
                raise DockerZeroRuntimeReconcilerV2Error(
                    "Docker data-root changed during recovery"
                )
            return None
        except DockerZeroRuntimeReconcilerV2Error as exc:
            if attempt == ready_attempts:
                errors.append(str(exc))
                break
            sleeper(1)
    return "; ".join(errors) or "Docker recovery did not become ready"


def reconcile_zero_runtime_docker_daemon_v2(
    *,
    lock_file: Path,
    admission_lock_file: Path,
    lock_owner_pid: int,
    runner: CommandRunner = _run,
    expected_root: Path = Path("/var/lib/docker"),
    ready_attempts: int = 30,
    total_timeout_seconds: int = 480,
    sleeper: Sleeper = time.sleep,
    clock: Clock = time.monotonic,
    lock_verifier: Optional[LockVerifier] = None,
) -> DockerZeroRuntimeReconcileResult:
    """Restart dockerd only after proving its entire container runtime is empty."""

    if ready_attempts < 1 or ready_attempts > 300:
        raise DockerZeroRuntimeReconcilerV2Error(
            "Docker readiness attempts must be between 1 and 300"
        )
    if total_timeout_seconds < 180 or total_timeout_seconds > 480:
        raise DockerZeroRuntimeReconcilerV2Error(
            "Docker reconciliation timeout must be between 180 and 480 seconds"
        )
    overall_deadline = clock() + total_timeout_seconds
    reconcile_deadline = overall_deadline - 120
    reconcile_runner = _deadline_runner(
        runner,
        deadline=reconcile_deadline,
        clock=clock,
    )
    verify_locks = lock_verifier or verify_docker_operation_locks_v2
    lock_arguments = {
        "lock_file": lock_file,
        "admission_lock_file": admission_lock_file,
        "owner_pid": lock_owner_pid,
    }
    verify_locks(**lock_arguments)
    before = _empty_runtime_snapshot(reconcile_runner, expected_root=expected_root)
    verify_locks(**lock_arguments)

    failure: Optional[DockerZeroRuntimeReconcilerV2Error] = None
    try:
        _checked_output(
            reconcile_runner,
            ["systemctl", "restart", "docker.service"],
            label="Docker daemon metadata reconciliation",
        )
        after: Optional[_EmptyRuntimeSnapshot] = None
        last_error: Optional[DockerZeroRuntimeReconcilerV2Error] = None
        for attempt in range(1, ready_attempts + 1):
            if clock() >= reconcile_deadline:
                last_error = DockerZeroRuntimeReconcilerV2Error(
                    "Docker reconciliation readiness deadline was exhausted"
                )
                break
            try:
                after = _empty_runtime_snapshot(
                    reconcile_runner,
                    expected_root=expected_root,
                )
                break
            except DockerZeroRuntimeReconcilerV2Error as exc:
                last_error = exc
                if attempt < ready_attempts:
                    sleeper(1)
        if after is None:
            raise DockerZeroRuntimeReconcilerV2Error(
                "Docker empty runtime did not recover after metadata reconciliation"
            ) from last_error
        if after.containerd_service != before.containerd_service:
            raise DockerZeroRuntimeReconcilerV2Error(
                "containerd identity changed during dockerd-only reconciliation"
            )
        if after.docker_service.invocation_id == before.docker_service.invocation_id:
            raise DockerZeroRuntimeReconcilerV2Error(
                "dockerd invocation identity did not change during reconciliation"
            )
        if after.docker_root != before.docker_root:
            raise DockerZeroRuntimeReconcilerV2Error(
                "Docker data-root changed during daemon reconciliation"
            )
        if after.docker_root_identity != before.docker_root_identity:
            raise DockerZeroRuntimeReconcilerV2Error(
                "Docker data-root inode changed during daemon reconciliation"
            )
        if after.image_ids != before.image_ids:
            raise DockerZeroRuntimeReconcilerV2Error(
                "Docker image identity changed during daemon reconciliation"
            )
        verify_locks(**lock_arguments)
    except DockerZeroRuntimeReconcilerV2Error as exc:
        failure = exc

    if failure is not None:
        try:
            verify_locks(**lock_arguments)
        except DockerZeroRuntimeReconcilerV2Error as exc:
            raise DockerZeroRuntimeReconcilerV2Error(
                f"{failure}; refusing Docker recovery after operation lock loss: {exc}"
            ) from failure
        recovery_runner = _deadline_runner(
            runner,
            deadline=overall_deadline,
            clock=clock,
        )
        recovery_error = _recover_docker_service(
            recovery_runner,
            expected_containerd=before.containerd_service,
            expected_root=expected_root,
            ready_attempts=ready_attempts,
            sleeper=sleeper,
            deadline=overall_deadline,
            clock=clock,
        )
        try:
            verify_locks(**lock_arguments)
        except DockerZeroRuntimeReconcilerV2Error as exc:
            lock_error = f"Docker operation lock verification failed: {exc}"
            recovery_error = (
                f"{recovery_error}; {lock_error}"
                if recovery_error
                else lock_error
            )
        if recovery_error:
            raise DockerZeroRuntimeReconcilerV2Error(
                f"{failure}; Docker recovery failed: {recovery_error}"
            ) from failure
        raise failure

    image_payload = json.dumps(
        before.image_ids,
        separators=(",", ":"),
    ).encode("utf-8")
    return DockerZeroRuntimeReconcileResult(
        docker_root=before.docker_root,
        image_count=len(before.image_ids),
        image_manifest_hash="sha256:"
        + hashlib.sha256(image_payload).hexdigest(),
        root_device=before.docker_root_identity.device,
        root_inode=before.docker_root_identity.inode,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-lock-file", type=Path, required=True)
    parser.add_argument("--docker-admission-lock-file", type=Path, required=True)
    parser.add_argument("--docker-lock-owner-pid", type=int, required=True)
    parser.add_argument("--ready-attempts", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=480)
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        raise SystemExit("Docker zero-runtime reconciliation must run as root")
    try:
        result = reconcile_zero_runtime_docker_daemon_v2(
            lock_file=args.docker_lock_file,
            admission_lock_file=args.docker_admission_lock_file,
            lock_owner_pid=args.docker_lock_owner_pid,
            ready_attempts=args.ready_attempts,
            total_timeout_seconds=args.timeout_seconds,
        )
    except DockerZeroRuntimeReconcilerV2Error as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
