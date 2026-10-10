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
import sys
from pathlib import Path
from typing import Sequence, Tuple

from lab_arena.trajectory import redact_text

RUNSC_CHECK_TIMEOUT_SECONDS = 5
RUNSC_CHECK_CLEANUP_SECONDS = 2
DEPENDENCY_INSTALLER_CHECK_TIMEOUT_SECONDS = 5
DEPENDENCY_INSTALLER_CHECK_CLEANUP_SECONDS = 2
DEFAULT_RUNNER_SOCKET_ROOT = Path("/tmp")
HOST_MEMORY_RESERVE_BYTES = 2 * 1024 ** 3
PER_SLOT_MEMORY_RESERVE_BYTES = 128 * 1024 ** 2
DEFAULT_SANDBOX_MEMORY_BYTES = 2 * 1024 ** 3
_REASONS = {
    "runtime_host_error": "the host cannot run the Arena sandbox",
    "runsc_path_invalid": "configure an absolute path to the installed runsc executable",
    "runsc_missing": "install runsc or correct its configured path",
    "runsc_not_executable": "runsc must be a regular executable file",
    "runsc_unusable": "the installed runsc could not execute; check its architecture and complete installation",
    "runsc_probe_timeout": "the installed runsc did not respond to its bounded version check",
    "runsc_probe_cleanup_failed": "the version check could not be stopped and reaped; inspect the host before retrying",
    "dependency_installer_unavailable": "install pip for the configured Arena Python interpreter",
    "dependency_installer_probe_timeout": "the pip readiness check did not finish; inspect the host before retrying",
    "dependency_installer_probe_cleanup_failed": "the pip readiness check could not be stopped and reaped; inspect the host before retrying",
    "unsupported_host": "Arena scoring requires Linux x86_64",
    "root_required": "Arena scoring requires the configured rootful service",
    "unsafe_work_directory": "runner directories must be real directories at a dedicated path",
    "work_directory_unwritable": "check runner directory ownership, permissions and available storage",
    "sandbox_launch_failed": "runsc exited before sandbox creation completed",
    "sandbox_launcher_signaled": "the host runsc process was terminated by a signal",
    "sandbox_startup_timeout": "runsc did not complete sandbox creation before the startup deadline",
    "parallel_memory_insufficient": "configured proxy slots exceed available memory; provide 2 GiB per sandbox plus host reserve",
    "parallel_memory_unavailable": "the host memory limit could not be verified",
}
_LOG_PATH = re.compile(r"/[A-Za-z0-9_./ -]{0,511}\Z")
MAX_LAUNCH_DIAGNOSTIC_CHARS = 2048
MAX_LAUNCH_DIAGNOSTIC_BYTES = 64 * 1024
_DIAGNOSTIC_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)[?#][^\s]*", re.IGNORECASE)
_DIAGNOSTIC_URL_AUTHORITY_RE = re.compile(r"(https?://)[^\s/@]+(?::[^\s/@]*)?@", re.IGNORECASE)
_DIAGNOSTIC_HEADER_RE = re.compile(r"(?im)(\b(?:cookie|set-cookie|authorization)\s*:\s*)[^\r\n]*")
_DIAGNOSTIC_CREDENTIAL_RE = re.compile(
    r"(?i)(\b(?:[a-z0-9_]*(?:api[_-]?key|token|secret|password|passwd)|"
    r"authorization|cookie|set-cookie)\b[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|(?:(?:bearer|basic)\s+)?[^\s,;]+)"
)


def _safe_launch_stderr(
    value: bytes, *, truncated: bool, secrets: Sequence[str]
) -> Tuple[str, bool]:
    # Capture is a bounded prefix. Drop a cut final line before redaction so
    # a partial credential value cannot outlive its label.
    truncated = truncated or len(value) > MAX_LAUNCH_DIAGNOSTIC_BYTES
    data = bytes(value[:MAX_LAUNCH_DIAGNOSTIC_BYTES])
    if truncated:
        data = data.rpartition(b"\n")[0]
    text = data.decode("utf-8", errors="replace")
    # Remove quoted values and full credential headers before the common
    # redactor, which can replace only the first word of a quoted value.
    text = _DIAGNOSTIC_HEADER_RE.sub(r"\1[redacted]", text)
    text = _DIAGNOSTIC_CREDENTIAL_RE.sub(r"\1[redacted]", text)
    text = redact_text(text, secrets=secrets)
    text = _DIAGNOSTIC_URL_QUERY_RE.sub(r"\1?[redacted]", text)
    text = _DIAGNOSTIC_URL_AUTHORITY_RE.sub(r"\1[redacted]@", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]+", " ", text)
    text = " ".join(text.split())
    return text[:MAX_LAUNCH_DIAGNOSTIC_CHARS], truncated or len(text) > MAX_LAUNCH_DIAGNOSTIC_CHARS


class ArenaRuntimeError(RuntimeError):
    """Base runtime failure; always fail the ICP attempt closed."""


class RuntimeHostError(ArenaRuntimeError):
    """A host failure with a closed, credential-free public diagnostic."""

    def __init__(
        self, message="", *, reason="runtime_host_error", runsc_path=None, work_dir=None,
        launch_exit_code=None, launch_timed_out=False, launch_stderr=b"",
        launch_stderr_truncated=False, diagnostic_secrets: Sequence[str] = (),
    ):
        # Preserve existing callers' exception text, but never log that text.
        super().__init__(
            message or _REASONS.get(reason, _REASONS["runtime_host_error"])
        )
        self.reason = reason if reason in _REASONS else "runtime_host_error"
        self.runsc_path = runsc_path
        self.work_dir = work_dir
        self.launch_exit_code = (
            launch_exit_code if isinstance(launch_exit_code, int)
            and not isinstance(launch_exit_code, bool) and -255 <= launch_exit_code <= 255
            else None
        )
        self.launch_timed_out = bool(launch_timed_out)
        self.launch_stderr, self.launch_stderr_truncated = _safe_launch_stderr(
            launch_stderr, truncated=bool(launch_stderr_truncated),
            secrets=diagnostic_secrets,
        )


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


def runtime_host_private_diagnostic(error: RuntimeHostError) -> str:
    """Include bounded redacted launcher detail only in private host logs."""
    diagnostic = runtime_host_diagnostic(error)
    if error.reason not in ("sandbox_launch_failed", "sandbox_launcher_signaled", "sandbox_startup_timeout"):
        return diagnostic
    return "%s launch_exit_code=%s launch_timed_out=%s launch_stderr_truncated=%s launch_stderr=%s" % (
        diagnostic, error.launch_exit_code, str(error.launch_timed_out).lower(),
        str(error.launch_stderr_truncated).lower(), json.dumps(error.launch_stderr),
    )


def require_linux_x86_64() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in (
        "x86_64",
        "amd64",
    ):
        raise RuntimeHostError(reason="unsupported_host")


def _available_parallel_memory(
    *,
    meminfo_path: Path,
    cgroup_root: Path,
    membership_path: Path,
) -> int:
    """Return memory available inside the tightest applicable host limit."""
    try:
        values = {parts[0].rstrip(":"): int(parts[1]) * 1024
                  for line in meminfo_path.read_text().splitlines()
                  if len(parts := line.split()) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:")}
        available = min(values["MemTotal"], values["MemAvailable"])
        # Normal service installations use the unified cgroup hierarchy. A
        # tighter systemd/container limit must take precedence over host RAM.
        memberships = membership_path.read_text().splitlines()
        for membership in memberships:
            _hierarchy, controllers, group = membership.split(":", 2)
            unified = membership.startswith("0::")
            if not unified and "memory" not in controllers.split(","):
                continue
            memory_root = cgroup_root if unified else cgroup_root / "memory"
            limit_name = "memory.max" if unified else "memory.limit_in_bytes"
            used_name = "memory.current" if unified else "memory.usage_in_bytes"
            relative = Path(group.lstrip("/"))
            if ".." in relative.parts:
                raise ValueError("invalid cgroup membership")
            directory = memory_root / relative
            while True:
                limit_path = directory / limit_name
                if limit_path.is_file():
                    limit = limit_path.read_text().strip()
                    if limit != "max":
                        used = int((directory / used_name).read_text())
                        available = min(available, max(0, int(limit) - used))
                if directory == memory_root:
                    break
                directory = directory.parent
    except (OSError, ValueError, KeyError):
        raise RuntimeHostError(reason="parallel_memory_unavailable") from None
    return available


def parallel_memory_capacity(
    slot_ceiling: int,
    sandbox_bytes: int,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    membership_path: Path = Path("/proc/self/cgroup"),
) -> int:
    """Return the safe slot count up to ``slot_ceiling``, or fail closed."""
    available = _available_parallel_memory(
        meminfo_path=meminfo_path,
        cgroup_root=cgroup_root,
        membership_path=membership_path,
    )
    try:
        if slot_ceiling < 1 or sandbox_bytes < 1:
            raise ValueError("invalid sandbox memory limits")
        per_slot = sandbox_bytes + PER_SLOT_MEMORY_RESERVE_BYTES
        supported = (available - HOST_MEMORY_RESERVE_BYTES) // per_slot
    except (TypeError, ValueError, ZeroDivisionError):
        raise RuntimeHostError(reason="parallel_memory_unavailable") from None
    if supported < 1:
        raise RuntimeHostError(reason="parallel_memory_insufficient")
    return min(slot_ceiling, supported)


def require_parallel_memory(
    slots: int,
    sandbox_bytes: int,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    membership_path: Path = Path("/proc/self/cgroup"),
) -> None:
    """Refuse an overcommitted fleet before claiming work; weights stay independent."""
    supported = parallel_memory_capacity(
        slots,
        sandbox_bytes,
        meminfo_path=meminfo_path,
        cgroup_root=cgroup_root,
        membership_path=membership_path,
    )
    if supported < slots:
        raise RuntimeHostError(reason="parallel_memory_insufficient")


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


def check_dependency_installer() -> None:
    """Prove this interpreter can launch pip before the runner claims work."""

    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-m", "pip", "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
            start_new_session=True,
        )
    except OSError:
        raise RuntimeHostError(reason="dependency_installer_unavailable") from None
    try:
        returncode = process.wait(timeout=DEPENDENCY_INSTALLER_CHECK_TIMEOUT_SECONDS)
    except BaseException as exc:
        try:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=DEPENDENCY_INSTALLER_CHECK_CLEANUP_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeHostError(
                reason="dependency_installer_probe_cleanup_failed"
            ) from None
        if isinstance(exc, subprocess.TimeoutExpired):
            raise RuntimeHostError(
                reason="dependency_installer_probe_timeout"
            ) from None
        raise
    if returncode != 0:
        raise RuntimeHostError(reason="dependency_installer_unavailable")


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
    check_dependency_installer()
    prepare_runner_directories(work_dir)
    check_runner_socket_directory()
