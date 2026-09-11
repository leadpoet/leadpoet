"""Local scoring readiness, independent of wallets, chain clients and providers."""

from __future__ import annotations

import errno
import json
import os
import platform
import re
import secrets
import signal
import stat
import subprocess
from pathlib import Path

RUNSC_CHECK_TIMEOUT_SECONDS = 5
RUNSC_CHECK_CLEANUP_SECONDS = 2
DEFAULT_RUNNER_SOCKET_ROOT = Path("/tmp")
_REASONS = {
    "runtime_host_error": "the host cannot run the Arena sandbox",
    "runsc_path_invalid": "configure an absolute path to the installed runsc executable",
    "runsc_missing": "install runsc or correct its configured path",
    "runsc_not_executable": "runsc must be a regular executable file",
    "runsc_unusable": "the installed runsc could not execute; check its architecture and complete installation",
    "runsc_probe_timeout": "the installed runsc did not respond to its bounded version check",
    "runsc_probe_cleanup_failed": "the version check could not be stopped and reaped; inspect the host before retrying",
    "unsupported_host": "Arena scoring requires Linux x86_64",
    "root_required": "Arena scoring requires the configured rootful service",
    "unsafe_work_directory": "runner directories must be real directories at a dedicated path",
    "work_directory_unwritable": "check runner directory ownership, permissions and available storage",
    "sandbox_launch_failed": "runsc exited before sandbox creation completed",
}
_LOG_PATH = re.compile(r"/[A-Za-z0-9_./ -]{0,511}\Z")


class ArenaRuntimeError(RuntimeError):
    """Base runtime failure; always fail the ICP attempt closed."""


class RuntimeHostError(ArenaRuntimeError):
    """A host failure with a closed, credential-free public diagnostic."""

    def __init__(
        self, message="", *, reason="runtime_host_error", runsc_path=None, work_dir=None
    ):
        # Preserve existing callers' exception text, but never log that text.
        super().__init__(
            message or _REASONS.get(reason, _REASONS["runtime_host_error"])
        )
        self.reason = reason if reason in _REASONS else "runtime_host_error"
        self.runsc_path = runsc_path
        self.work_dir = work_dir


def scoring_host_details(*, runsc_path=None, work_dir=None) -> str:
    """Only bounded local paths are public; never format arbitrary metadata."""
    fields = []
    for name, path in (("runsc_path", runsc_path), ("work_dir", work_dir)):
        if path is not None:
            value = str(path)
            if _LOG_PATH.fullmatch(value):
                fields.append("%s=%s" % (name, json.dumps(value)))
    return " ".join(fields)


def runtime_host_diagnostic(error: RuntimeHostError) -> str:
    reason = error.reason if error.reason in _REASONS else "runtime_host_error"
    details = scoring_host_details(runsc_path=error.runsc_path, work_dir=error.work_dir)
    return "reason=%s%s hint=%s" % (
        reason,
        " " + details if details else "",
        json.dumps(_REASONS[reason]),
    )


def require_linux_x86_64() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in (
        "x86_64",
        "amd64",
    ):
        raise RuntimeHostError(reason="unsupported_host")


def require_rootful_runtime() -> None:
    if getattr(os, "geteuid", lambda: -1)() != 0:
        raise RuntimeHostError(reason="root_required")


def require_runsc_executable(path: Path) -> Path:
    """Check the explicitly selected binary; never search PATH or install one."""
    binary = Path(path)
    if not binary.is_absolute():
        raise RuntimeHostError(reason="runsc_path_invalid")
    try:
        metadata = binary.stat()
    except ValueError:
        raise RuntimeHostError(reason="runsc_path_invalid") from None
    except FileNotFoundError:
        raise RuntimeHostError(reason="runsc_missing", runsc_path=binary) from None
    except OSError:
        raise RuntimeHostError(
            reason="runsc_not_executable", runsc_path=binary
        ) from None
    if not stat.S_ISREG(metadata.st_mode) or not os.access(binary, os.X_OK):
        raise RuntimeHostError(reason="runsc_not_executable", runsc_path=binary)
    return binary


def check_runsc_launch(binary: Path) -> None:
    """A bounded version check proves execution, not sandbox capability."""
    try:
        process = subprocess.Popen(
            [str(binary), "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
            start_new_session=True,
        )
    except OSError as exc:
        reason = (
            "runsc_not_executable"
            if exc.errno in (errno.EACCES, errno.EPERM)
            else "runsc_unusable"
        )
        raise RuntimeHostError(reason=reason, runsc_path=binary) from None
    try:
        returncode = process.wait(timeout=RUNSC_CHECK_TIMEOUT_SECONDS)
    except BaseException as exc:
        # A broken installation may spawn helpers before hanging. Kill the
        # entire task-owned group, then reap the version process on all exits.
        try:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=RUNSC_CHECK_CLEANUP_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeHostError(
                reason="runsc_probe_cleanup_failed", runsc_path=binary
            ) from None
        if isinstance(exc, subprocess.TimeoutExpired):
            raise RuntimeHostError(
                reason="runsc_probe_timeout", runsc_path=binary
            ) from None
        raise
    if returncode != 0:
        raise RuntimeHostError(reason="runsc_unusable", runsc_path=binary)


def _open_directory_tree(path: Path, *, create: bool) -> int:
    """Walk from / using no-follow opens, including every parent component."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for name in path.parts[1:]:
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass  # The no-follow directory open checks existing types.
            child = os.open(name, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _probe_directory_write(descriptor: int) -> None:
    # Use directory descriptors so a renamed path cannot redirect the probe.
    probe_name = ".arena-readiness-" + secrets.token_hex(12)
    probe = os.open(
        probe_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=descriptor,
    )
    try:
        os.write(probe, b"arena-readiness\n")
    finally:
        os.close(probe)
        os.unlink(probe_name, dir_fd=descriptor)


def _directory_failure(exc: OSError, directory: Path) -> RuntimeHostError:
    reason = (
        "unsafe_work_directory"
        if exc.errno in (errno.EEXIST, errno.ENOTDIR, errno.ELOOP)
        else "work_directory_unwritable"
    )
    return RuntimeHostError(reason=reason, work_dir=directory)


def prepare_runner_directories(work_dir: Path) -> None:
    """Check every mutable runner directory without loading or evicting caches."""
    root = Path(work_dir).absolute()
    if (
        str(root)
        in (
            "/",
            "/home",
            "/root",
            "/var",
            "/var/lib",
            "/tmp",
            "/var/tmp",
            "/usr",
            "/etc",
            "/opt",
        )
        or ".." in root.parts
        or "\x00" in str(root)
    ):
        raise RuntimeHostError(reason="unsafe_work_directory", work_dir=root)
    directory = root
    root_descriptor = None
    try:
        root_descriptor = _open_directory_tree(root, create=True)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        for name in (None, "sandboxes", "runs", "images", "sources"):
            directory = root if name is None else root / name
            if name is not None:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=root_descriptor)
                except FileExistsError:
                    pass  # The no-follow directory open below checks its type.
            descriptor = (
                os.dup(root_descriptor)
                if name is None
                else os.open(name, flags, dir_fd=root_descriptor)
            )
            try:
                if name in ("sandboxes", "runs"):
                    os.fchmod(descriptor, 0o700)
                _probe_directory_write(descriptor)
            finally:
                os.close(descriptor)
    except OSError as exc:
        raise _directory_failure(exc, directory) from None
    except ValueError:
        raise RuntimeHostError(reason="unsafe_work_directory") from None
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)


def check_runner_socket_directory() -> None:
    """Probe the fixed OS temporary directory without changing its permissions."""
    directory = DEFAULT_RUNNER_SOCKET_ROOT
    descriptor = None
    try:
        # /tmp is a trusted OS alias on some hosts (including macOS test hosts).
        # Configurable work paths above never get this symlink-resolution exception.
        resolved = directory.resolve(strict=True)
        descriptor = _open_directory_tree(resolved, create=False)
        _probe_directory_write(descriptor)
    except OSError as exc:
        raise _directory_failure(exc, directory) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def prepare_scoring_host(runsc_path: Path, work_dir: Path) -> None:
    """Shared setup for scoring startup and the offline host-readiness command."""
    require_linux_x86_64()
    binary = require_runsc_executable(runsc_path)
    require_rootful_runtime()
    check_runsc_launch(binary)
    prepare_runner_directories(work_dir)
    check_runner_socket_directory()
