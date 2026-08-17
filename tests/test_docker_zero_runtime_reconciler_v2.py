from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import pytest

from validator_tee.host.docker_zero_runtime_reconciler_v2 import (
    DockerZeroRuntimeReconcilerV2Error,
    _RootIdentity,
    _require_parent_fd_lock,
    reconcile_zero_runtime_docker_daemon_v2,
    verify_docker_operation_locks_v2,
)


class _Runtime:
    def __init__(
        self,
        root: Path,
        *,
        docker_containers: int = 0,
        containerd_containers: int = 0,
        containerd_tasks: int = 0,
        moby_shims: int = 0,
        change_images: bool = False,
        change_root_inode: bool = False,
        change_containerd: bool = False,
        keep_docker_invocation: bool = False,
        post_snapshot_failures: int = 0,
        restart_failure: bool = False,
        shim_output: str | None = None,
    ) -> None:
        self.root = root
        self.docker_containers = docker_containers
        self.containerd_containers = containerd_containers
        self.containerd_tasks = containerd_tasks
        self.moby_shims = moby_shims
        self.change_images = change_images
        self.change_root_inode = change_root_inode
        self.change_containerd = change_containerd
        self.keep_docker_invocation = keep_docker_invocation
        self.post_snapshot_failures = post_snapshot_failures
        self.restart_failure = restart_failure
        self.shim_output = shim_output
        self.restart_count = 0
        self.start_count = 0
        self.commands: list[tuple[str, ...]] = []
        self.image_ids = (
            "sha256:" + hashlib.sha256(b"image-one").hexdigest(),
            "sha256:" + hashlib.sha256(b"image-two").hexdigest(),
        )
        self.container_id = hashlib.sha256(b"container").hexdigest()

    @staticmethod
    def _ids(identity: str, count: int) -> str:
        return "" if count == 0 else "\n".join([identity] * count) + "\n"

    def _service(self, service: str) -> str:
        if service == "docker.service":
            pid = 1001 + self.restart_count
            invocation = "1" * 32
            if self.restart_count and not self.keep_docker_invocation:
                invocation = "2" * 32
        else:
            pid = 2001 + int(self.restart_count > 0 and self.change_containerd)
            invocation = "3" * 32
            if self.restart_count and self.change_containerd:
                invocation = "4" * 32
        return (
            "LoadState=loaded\n"
            "ActiveState=active\n"
            "SubState=running\n"
            f"MainPID={pid}\n"
            f"InvocationID={invocation}\n"
        )

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        values = tuple(command)
        self.commands.append(values)
        stdout = ""
        stderr = ""
        returncode = 0
        if values[:3] == ("systemctl", "show", "docker.service"):
            stdout = self._service("docker.service")
        elif values[:3] == ("systemctl", "show", "containerd.service"):
            stdout = self._service("containerd.service")
        elif values == ("docker", "info", "--format", "{{.DockerRootDir}}"):
            if self.restart_count and self.post_snapshot_failures:
                self.post_snapshot_failures -= 1
                returncode = 1
                stderr = "daemon not ready"
            else:
                stdout = str(self.root) + "\n"
        elif values == ("docker", "ps", "-aq", "--no-trunc"):
            stdout = self._ids(self.container_id, self.docker_containers)
        elif values == ("ctr", "-n", "moby", "containers", "list", "-q"):
            stdout = self._ids(self.container_id, self.containerd_containers)
        elif values == ("ctr", "-n", "moby", "tasks", "list", "-q"):
            stdout = self._ids(self.container_id, self.containerd_tasks)
        elif values == (
            "pgrep",
            "-fc",
            "^/usr/bin/containerd-shim-runc-v2 -namespace moby ",
        ):
            stdout = (
                self.shim_output
                if self.shim_output is not None
                else f"{self.moby_shims}\n"
            )
            returncode = 0 if self.moby_shims else 1
        elif values == ("docker", "image", "ls", "-aq", "--no-trunc"):
            images = self.image_ids
            if self.restart_count and self.change_images:
                images = images[:1]
            # Duplicate one ID to model a multi-tag image.
            stdout = "\n".join([*images, images[0]]) + "\n"
        elif values == ("systemctl", "restart", "docker.service"):
            self.restart_count += 1
            if self.change_root_inode:
                self.root.rmdir()
                self.root.mkdir(mode=0o700)
            if self.restart_failure:
                returncode = 1
                stderr = "restart failed"
        elif values == ("systemctl", "start", "docker.service"):
            self.start_count += 1
        else:
            raise AssertionError(f"unexpected command: {values}")
        return subprocess.CompletedProcess(
            list(values),
            returncode,
            stdout=stdout,
            stderr=stderr,
        )


class _LockVerifier:
    def __init__(self, *, fail_on_call: int = 0) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, **_kwargs: object) -> None:
        self.calls += 1
        if self.fail_on_call and self.calls >= self.fail_on_call:
            raise DockerZeroRuntimeReconcilerV2Error("operation locks were lost")


def _fdinfo_payload(path: Path, *, owner_pid: int = 0, include_lock: bool = True) -> str:
    metadata = path.stat()
    payload = (
        "pos:\t0\n"
        "flags:\t0100002\n"
        "mnt_id:\t1\n"
        f"ino:\t{metadata.st_ino}\n"
    )
    if include_lock:
        payload += (
            "lock:\t1: FLOCK  ADVISORY  WRITE "
            f"{owner_pid} {os.major(metadata.st_dev):02x}:"
            f"{os.minor(metadata.st_dev):02x}:{metadata.st_ino} 0 EOF\n"
        )
    return payload


def _write_synthetic_proc_owner(
    proc_root: Path,
    *,
    owner_pid: int,
    resource: Path,
    admission: Path,
    include_locks: bool = True,
) -> None:
    owner_root = proc_root / str(owner_pid)
    fd_root = owner_root / "fd"
    fdinfo_root = owner_root / "fdinfo"
    fd_root.mkdir(parents=True)
    fdinfo_root.mkdir()
    (fd_root / "7").symlink_to(resource)
    (fd_root / "8").symlink_to(admission)
    (fdinfo_root / "7").write_text(
        _fdinfo_payload(resource, include_lock=include_locks),
        encoding="ascii",
    )
    (fdinfo_root / "8").write_text(
        _fdinfo_payload(admission, include_lock=include_locks),
        encoding="ascii",
    )


def _reconcile(
    tmp_path: Path,
    runtime: _Runtime,
    *,
    verifier: _LockVerifier | None = None,
    ready_attempts: int = 3,
    total_timeout_seconds: int = 480,
    sleeper=lambda _seconds: None,
    clock=None,
):
    return reconcile_zero_runtime_docker_daemon_v2(
        lock_file=tmp_path / "operation.lock",
        admission_lock_file=tmp_path / "admission.lock",
        lock_owner_pid=123,
        runner=runtime,
        expected_root=runtime.root,
        ready_attempts=ready_attempts,
        total_timeout_seconds=total_timeout_seconds,
        sleeper=sleeper,
        **({"clock": clock} if clock is not None else {}),
        lock_verifier=verifier or _LockVerifier(),
    )


def test_reconcile_restarts_only_dockerd_and_preserves_images_and_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "docker"
    root.mkdir(mode=0o700)
    runtime = _Runtime(root)
    verifier = _LockVerifier()

    result = _reconcile(tmp_path, runtime, verifier=verifier)

    assert result.docker_root == str(root)
    assert result.image_count == 2
    assert runtime.restart_count == 1
    assert runtime.start_count == 0
    assert verifier.calls == 3
    assert ("systemctl", "restart", "docker.service") in runtime.commands
    assert not any(
        "containerd.service" in command and command[:2] in {
            ("systemctl", "restart"),
            ("systemctl", "start"),
        }
        for command in runtime.commands
    )
    assert not any(
        command[:2] == ("docker", "image")
        and command != ("docker", "image", "ls", "-aq", "--no-trunc")
        for command in runtime.commands
    )
    assert not any(
        command and command[0] in {"rm", "umount"}
        or "prune" in command
        for command in runtime.commands
    )
    assert not any(
        "daemon.json" in value or "live-restore" in value
        for command in runtime.commands
        for value in command
    )


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("docker_containers", "Docker container inventory is nonempty"),
        ("containerd_containers", "containerd container inventory is nonempty"),
        ("containerd_tasks", "containerd task inventory is nonempty"),
        ("moby_shims", "moby shims are present"),
    ],
)
def test_reconcile_refuses_every_nonempty_runtime_signal(
    tmp_path: Path,
    setting: str,
    message: str,
) -> None:
    root = tmp_path / "docker"
    root.mkdir(mode=0o700)
    runtime = _Runtime(root, **{setting: 1})

    with pytest.raises(DockerZeroRuntimeReconcilerV2Error, match=message):
        _reconcile(tmp_path, runtime)

    assert runtime.restart_count == 0
    assert runtime.start_count == 0


@pytest.mark.parametrize(
    ("runtime_kwargs", "message"),
    [
        ({"change_images": True}, "image identity changed"),
        ({"change_root_inode": True}, "data-root inode changed"),
        ({"change_containerd": True}, "containerd identity changed"),
        ({"keep_docker_invocation": True}, "invocation identity did not change"),
        ({"restart_failure": True}, "metadata reconciliation failed"),
    ],
)
def test_post_restart_identity_failures_are_terminal_but_recover_docker(
    tmp_path: Path,
    runtime_kwargs: dict[str, bool],
    message: str,
) -> None:
    root = tmp_path / "docker"
    root.mkdir(mode=0o700)
    runtime = _Runtime(root, **runtime_kwargs)

    with pytest.raises(DockerZeroRuntimeReconcilerV2Error, match=message):
        _reconcile(tmp_path, runtime)

    assert runtime.restart_count == 1
    assert runtime.start_count == 1


def test_reconcile_rejects_noncanonical_shim_count_before_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "docker"
    root.mkdir(mode=0o700)
    runtime = _Runtime(root, shim_output=" 0\n")

    with pytest.raises(
        DockerZeroRuntimeReconcilerV2Error,
        match="shim inventory is malformed",
    ):
        _reconcile(tmp_path, runtime)

    assert runtime.restart_count == 0
    assert runtime.start_count == 0


def test_reconcile_fails_closed_when_parent_loses_lock_after_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "docker"
    root.mkdir(mode=0o700)
    runtime = _Runtime(root)
    verifier = _LockVerifier(fail_on_call=3)

    with pytest.raises(DockerZeroRuntimeReconcilerV2Error, match="locks were lost"):
        _reconcile(tmp_path, runtime, verifier=verifier)

    assert runtime.restart_count == 1
    assert runtime.start_count == 0


def test_reconcile_bounds_readiness_and_reports_failed_recovery(tmp_path: Path) -> None:
    root = tmp_path / "docker"
    root.mkdir(mode=0o700)
    runtime = _Runtime(root, post_snapshot_failures=20)

    with pytest.raises(
        DockerZeroRuntimeReconcilerV2Error,
        match="empty runtime did not recover.*Docker recovery failed",
    ):
        _reconcile(tmp_path, runtime, ready_attempts=2)

    assert runtime.restart_count == 1
    assert runtime.start_count == 1


def test_reconcile_reserves_a_bounded_recovery_window(tmp_path: Path) -> None:
    root = tmp_path / "docker"
    root.mkdir(mode=0o700)
    runtime = _Runtime(root, post_snapshot_failures=300)
    now = [0.0]

    def clock() -> float:
        return now[0]

    def advance(_seconds: float) -> None:
        now[0] += 30

    with pytest.raises(
        DockerZeroRuntimeReconcilerV2Error,
        match="empty runtime did not recover.*Docker recovery failed",
    ):
        _reconcile(
            tmp_path,
            runtime,
            ready_attempts=300,
            total_timeout_seconds=180,
            sleeper=advance,
            clock=clock,
        )

    assert runtime.restart_count == 1
    assert runtime.start_count == 1
    assert now[0] <= 180
    assert len(runtime.commands) < 50


def test_exact_parent_resource_and_admission_locks_are_required(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "operation.lock"
    admission = tmp_path / "admission.lock"
    resource.touch(mode=0o600)
    admission.touch(mode=0o600)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, os, sys; "
                "a=os.open(sys.argv[1], os.O_RDWR); "
                "b=os.open(sys.argv[2], os.O_RDWR); "
                "os.dup2(a, 7); os.dup2(b, 8); "
                "os.close(a); os.close(b); "
                "fcntl.flock(7, fcntl.LOCK_EX); "
                "fcntl.flock(8, fcntl.LOCK_EX); "
                "print('ready', flush=True); sys.stdin.read(1)"
            ),
            str(resource),
            str(admission),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "ready"
    proc_root = tmp_path / "proc"
    _write_synthetic_proc_owner(
        proc_root,
        owner_pid=holder.pid,
        resource=resource,
        admission=admission,
    )
    wrong_admission = tmp_path / "wrong-admission.lock"
    wrong_admission.touch(mode=0o600)
    try:
        verify_docker_operation_locks_v2(
            lock_file=resource,
            admission_lock_file=admission,
            owner_pid=holder.pid,
            proc_root=proc_root,
        )
        with pytest.raises(
            DockerZeroRuntimeReconcilerV2Error,
            match="not owned by the declared parent",
        ):
            verify_docker_operation_locks_v2(
                lock_file=resource,
                admission_lock_file=wrong_admission,
                owner_pid=holder.pid,
                proc_root=proc_root,
            )
    finally:
        if holder.stdin is not None:
            holder.stdin.write("x")
            holder.stdin.flush()
        holder.wait(timeout=5)

    with pytest.raises(DockerZeroRuntimeReconcilerV2Error, match="is not held"):
        verify_docker_operation_locks_v2(
            lock_file=resource,
            admission_lock_file=admission,
            owner_pid=holder.pid,
            proc_root=proc_root,
        )


def test_different_process_lock_does_not_authorize_an_unlocked_parent(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "operation.lock"
    admission = tmp_path / "admission.lock"
    resource.touch(mode=0o600)
    admission.touch(mode=0o600)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, os, sys; "
                "a=os.open(sys.argv[1], os.O_RDWR); "
                "b=os.open(sys.argv[2], os.O_RDWR); "
                "fcntl.flock(a, fcntl.LOCK_EX); "
                "fcntl.flock(b, fcntl.LOCK_EX); "
                "print('ready', flush=True); sys.stdin.read(1)"
            ),
            str(resource),
            str(admission),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "a=os.open(sys.argv[1], os.O_RDWR); "
                "b=os.open(sys.argv[2], os.O_RDWR); "
                "os.dup2(a, 7); os.dup2(b, 8); "
                "os.close(a); os.close(b); "
                "print('ready', flush=True); sys.stdin.read(1)"
            ),
            str(resource),
            str(admission),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert owner.stdout is not None
    assert holder.stdout.readline().strip() == "ready"
    assert owner.stdout.readline().strip() == "ready"
    proc_root = tmp_path / "proc"
    _write_synthetic_proc_owner(
        proc_root,
        owner_pid=owner.pid,
        resource=resource,
        admission=admission,
        include_locks=False,
    )
    try:
        with pytest.raises(
            DockerZeroRuntimeReconcilerV2Error,
            match="does not prove an exclusive whole-file FLOCK",
        ):
            verify_docker_operation_locks_v2(
                lock_file=resource,
                admission_lock_file=admission,
                owner_pid=owner.pid,
                proc_root=proc_root,
            )
    finally:
        for process in (owner, holder):
            if process.stdin is not None:
                process.stdin.write("x")
                process.stdin.flush()
            process.wait(timeout=5)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate",
        "shared",
        "wrong_type",
        "wrong_pid",
        "wrong_device",
        "wrong_fdinfo_inode",
        "wrong_lock_inode",
        "partial_range",
        "malformed",
        "oversized",
    ],
)
def test_parent_fdinfo_lock_proof_rejects_malformed_or_mismatched_records(
    tmp_path: Path,
    case: str,
) -> None:
    lock_file = tmp_path / "operation.lock"
    lock_file.touch(mode=0o600)
    metadata = lock_file.stat()
    owner_pid = 123
    payload = _fdinfo_payload(lock_file)
    if case == "missing":
        payload = _fdinfo_payload(lock_file, include_lock=False)
    elif case == "duplicate":
        payload += next(
            line + "\n" for line in payload.splitlines() if line.startswith("lock:")
        )
    elif case == "shared":
        payload = payload.replace("WRITE", "READ")
    elif case == "wrong_type":
        payload = payload.replace("FLOCK", "POSIX")
    elif case == "wrong_pid":
        payload = payload.replace("WRITE 0 ", "WRITE 9999 ")
    elif case == "wrong_device":
        payload = payload.replace(
            f"{os.major(metadata.st_dev):02x}:{os.minor(metadata.st_dev):02x}:",
            f"{os.major(metadata.st_dev) + 1:02x}:{os.minor(metadata.st_dev):02x}:",
        )
    elif case == "wrong_fdinfo_inode":
        payload = payload.replace(
            f"ino:\t{metadata.st_ino}\n",
            f"ino:\t{metadata.st_ino + 1}\n",
        )
    elif case == "wrong_lock_inode":
        payload = payload.replace(
            f":{metadata.st_ino} 0 EOF",
            f":{metadata.st_ino + 1} 0 EOF",
        )
    elif case == "partial_range":
        payload = payload.replace(" 0 EOF\n", " 1 EOF\n")
    elif case == "malformed":
        payload = payload.replace("lock:\t1:", "lock: 1:")
    elif case == "oversized":
        payload += "padding:\t" + ("x" * (64 * 1024)) + "\n"
    fdinfo = tmp_path / "proc" / str(owner_pid) / "fdinfo"
    fdinfo.mkdir(parents=True)
    (fdinfo / "7").write_text(payload, encoding="ascii")

    with pytest.raises(DockerZeroRuntimeReconcilerV2Error, match="fdinfo"):
        _require_parent_fd_lock(
            proc_root=tmp_path / "proc",
            owner_pid=owner_pid,
            descriptor=7,
            identity=_RootIdentity(
                device=metadata.st_dev,
                inode=metadata.st_ino,
            ),
            label="Docker operation lock",
        )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc fdinfo contract")
def test_real_linux_shell_flock_is_bound_to_exact_parent_fdinfo(
    tmp_path: Path,
) -> None:
    resource = tmp_path / "operation.lock"
    admission = tmp_path / "admission.lock"
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                "set -euo pipefail; "
                "exec 7>>\"$1\"; exec 8>>\"$2\"; "
                "flock -x 7; flock -x 8; "
                "printf 'ready\\n'; IFS= read -r _"
            ),
            "bash",
            str(resource),
            str(admission),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "ready"
    try:
        verify_docker_operation_locks_v2(
            lock_file=resource,
            admission_lock_file=admission,
            owner_pid=holder.pid,
        )
    finally:
        if holder.stdin is not None:
            holder.stdin.write("x\n")
            holder.stdin.flush()
        holder.wait(timeout=5)
