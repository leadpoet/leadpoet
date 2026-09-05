"""Lab Arena sandbox runtime (labarena.md section 9.2).

One fresh gVisor sandbox per ICP: an executable runsc on a Linux x86_64 host,
cgroup CPU, memory, and pid limits, a
read-only model rootfs, uid/gid 65534, no new privileges, no network
interface (``runsc --network=none``), a bounded writable ``/output`` backed
by a size-bound host tmpfs, and exactly one Unix socket bind-mounted at
``/run/lab_arena/worker.sock``. The OCI document mirrors the enclave's proven
shape in ``gateway/tee/model_sandbox_v2.py`` without importing it.

Differences from the enclave, on purpose: the platform is ``systrap`` (the
enclave pins ``ptrace``; systrap is gVisor's current default and needs no
ptrace permission), no ``network`` namespace is listed in the OCI document
because ``--network=none`` is the isolation authority, and the output
directory is a host tmpfs so the worker can read ``companies.json`` after the
sandbox exits while its size stays bounded.

Host contract: Linux x86_64, root (rootful runsc and tmpfs mounts), and an
executable runsc. ``RunscRuntime`` verifies that at
construction and fails closed; ``run_sandbox`` is the pure orchestration with
injectable process runner, clock, and sleep so it is testable anywhere.
Defense in depth: the OCI seccomp rule allows ``socket`` only for AF_UNIX,
matching the enclave; runsc may ignore OCI seccomp, which is why network
isolation never relies on it.
"""

from __future__ import annotations

import json
import os
import platform
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from lab_arena import contracts

SANDBOX_MODEL_DIR = "/model"
SANDBOX_AGENT_DIR = "/agent"
SANDBOX_AGENT_SOURCE_DIR = SANDBOX_AGENT_DIR + "/source"
SANDBOX_AGENT_DEPENDENCY_DIR = SANDBOX_AGENT_DIR + "/deps"
SANDBOX_AGENT_ENTRYPOINT_PATH = SANDBOX_AGENT_DIR + "/entrypoint.py"
SANDBOX_INPUT_DIR = "/input"
SANDBOX_OUTPUT_DIR = "/output"
SANDBOX_SOCKET_DIR = "/run/lab_arena"
SANDBOX_SOCKET_NAME = "worker.sock"
SANDBOX_SOCKET_PATH = SANDBOX_SOCKET_DIR + "/" + SANDBOX_SOCKET_NAME
INPUT_FILE_NAME = "icp.json"
OUTPUT_FILE_NAME = "companies.json"
SANDBOX_INPUT_PATH = SANDBOX_INPUT_DIR + "/" + INPUT_FILE_NAME
SANDBOX_OUTPUT_PATH = SANDBOX_OUTPUT_DIR + "/" + OUTPUT_FILE_NAME
SANDBOX_HOSTNAME = "leadpoet-lab-arena"
# Isolated mode keeps the trusted scorer image's /model/sitecustomize.py from
# changing submitted source execution.  The host-owned entrypoint explicitly
# adds only the admitted source and dependency mounts after Python starts.
# -I ignores PYTHON* environment settings, so keep log flushing and read-only
# import behavior explicit. LAB_ARENA_RANDOM_SEED remains available to the agent.
AGENT_ENTRY_COMMAND = ("python3", "-I", "-u", "-B", SANDBOX_AGENT_ENTRYPOINT_PATH)
AGENT_WORKING_DIR = SANDBOX_AGENT_SOURCE_DIR
SCORER_ENTRY_COMMAND = ("python3", "/model/scorer_entrypoint.py")
SCORER_WORKING_DIR = "/model"

MAX_OUTPUT_BYTES = 512 * 1024
MAX_LOG_BYTES = 64 * 1024
TIMEOUT_GRACE_SECONDS = 10
STOP_GRACE_SECONDS = 5.0
CLEANUP_COMMAND_TIMEOUT_SECONDS = 30
OUTPUT_TMPFS_BYTES = 64 * 1024 * 1024
TMP_TMPFS_BYTES = 256 * 1024 * 1024
SANDBOX_UID = 65534
SANDBOX_GID = 65534
DEFAULT_CPU_QUOTA = 100_000
DEFAULT_CPU_PERIOD = 100_000
DEFAULT_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_PIDS_LIMIT = 256
DEFAULT_PLATFORM = "systrap"
RUNSC_PLATFORMS = ("systrap", "ptrace", "kvm")
LINUX_AF_UNIX = 1
PIPE_CHUNK_BYTES = 65536
PROCESS_ENV: Mapping[str, str] = MappingProxyType({"PATH": "/usr/local/bin:/usr/bin:/bin"})
PROVIDER_BASE_URLS: Mapping[str, str] = MappingProxyType(
    {
        "SCRAPINGDOG_BASE_URL": "https://api.scrapingdog.com",
        "DEEPLINE_BASE_URL": "https://code.deepline.com",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
    }
)

_SANDBOX_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

ProcessRunner = Callable[..., Any]


class ArenaRuntimeError(RuntimeError):
    """Base runtime failure; always fail the ICP attempt closed."""


class RuntimeHostError(ArenaRuntimeError):
    """The host cannot run the Arena runtime."""


class SandboxSpecError(ArenaRuntimeError):
    """A sandbox specification is invalid."""


class SandboxOutputError(ArenaRuntimeError):
    """The model output is not a bounded regular file."""


class SandboxCleanupError(ArenaRuntimeError):
    """Sandbox, mount, or bundle cleanup did not complete."""


# ---------------------------------------------------------------------------
# Runtime host
# ---------------------------------------------------------------------------


def require_runsc_executable(path: Path) -> Path:
    """Require a regular executable runsc file without pinning its identity."""

    binary = Path(path)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeHostError("runsc executable is unavailable")
    return binary


def require_linux_x86_64() -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Linux" or machine not in ("x86_64", "amd64"):
        raise RuntimeHostError("Arena runtime requires Linux x86_64 (host is %s %s)" % (system, machine))


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeConfig:
    runsc_path: Path
    work_dir: Path
    platform: str = DEFAULT_PLATFORM
    cleanup_timeout_seconds: int = CLEANUP_COMMAND_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.platform not in RUNSC_PLATFORMS:
            raise SandboxSpecError("runsc platform is invalid")
        if self.cleanup_timeout_seconds <= 0:
            raise SandboxSpecError("cleanup timeout is invalid")
        object.__setattr__(self, "runsc_path", Path(self.runsc_path))
        object.__setattr__(self, "work_dir", Path(self.work_dir))


@dataclass(frozen=True)
class SandboxSpec:
    """One ICP execution: paths are host paths, limits are cgroup values."""

    sandbox_id: str
    rootfs_path: Path
    input_dir: Path
    output_dir: Path
    socket_path: Path
    # A host-selected command. Miner bundles always use AGENT_ENTRY_COMMAND;
    # the trusted judge uses SCORER_ENTRY_COMMAND.
    entry_command: Tuple[str, ...]
    evaluation_date: str
    random_seed: int
    # Execute assignments bind the admitted source, optional installed wheels,
    # and the host-owned Python entrypoint. Scoring assignments leave these
    # unset and use only the trusted scorer rootfs.
    source_dir: Optional[Path] = None
    dependency_dir: Optional[Path] = None
    agent_entrypoint_path: Optional[Path] = None
    working_dir: str = ""
    cpu_quota: int = DEFAULT_CPU_QUOTA
    cpu_period: int = DEFAULT_CPU_PERIOD
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES
    pids_limit: int = DEFAULT_PIDS_LIMIT
    wall_clock_seconds: int = contracts.ICP_WALL_CLOCK_SECONDS
    uid: int = SANDBOX_UID
    gid: int = SANDBOX_GID
    output_tmpfs_bytes: int = OUTPUT_TMPFS_BYTES
    tmp_tmpfs_bytes: int = TMP_TMPFS_BYTES
    # Additional fixed environment (the scorer image's trusted mode and the
    # signed scorer policy bindings); never overrides the model environment.
    extra_environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _SANDBOX_ID_RE.match(str(self.sandbox_id)):
            raise SandboxSpecError("sandbox id is invalid")
        extra = dict(self.extra_environment or {})
        for name, value in extra.items():
            if not _ENV_NAME_RE.match(str(name)) or not isinstance(value, str) or len(value) > 4096 or any(ord(ch) < 32 for ch in value):
                raise SandboxSpecError("extra environment entry is invalid")
        object.__setattr__(self, "extra_environment", MappingProxyType(extra))
        for name in ("rootfs_path", "input_dir", "output_dir", "socket_path"):
            value = Path(getattr(self, name))
            if not value.is_absolute():
                raise SandboxSpecError("%s must be absolute" % name)
            object.__setattr__(self, name, value)
        agent_paths = (
            self.source_dir,
            self.dependency_dir,
            self.agent_entrypoint_path,
        )
        if any(value is not None for value in agent_paths) and not all(
            value is not None for value in agent_paths
        ):
            raise SandboxSpecError("agent source mounts must be supplied together")
        for name in ("source_dir", "dependency_dir", "agent_entrypoint_path"):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = Path(raw)
            if not value.is_absolute():
                raise SandboxSpecError("%s must be absolute" % name)
            object.__setattr__(self, name, value)
        if self.socket_path.name != SANDBOX_SOCKET_NAME:
            raise SandboxSpecError("worker socket must be named %s" % SANDBOX_SOCKET_NAME)
        command = tuple(self.entry_command) if isinstance(self.entry_command, (list, tuple)) else ()
        if not command or len(command) > 64 or any(not isinstance(item, str) or not item or len(item) > 4096 or "\x00" in item or "\n" in item for item in command):
            raise SandboxSpecError("entry command is invalid")
        object.__setattr__(self, "entry_command", command)
        if command == AGENT_ENTRY_COMMAND and not all(
            value is not None for value in agent_paths
        ):
            raise SandboxSpecError("agent source mounts are required")
        working_dir = str(self.working_dir or "")
        if working_dir and (not working_dir.startswith("/") or len(working_dir) > 4096 or "\x00" in working_dir or "\n" in working_dir):
            raise SandboxSpecError("working directory is invalid")
        object.__setattr__(self, "working_dir", working_dir)
        if not _DATE_RE.match(str(self.evaluation_date)):
            raise SandboxSpecError("evaluation date is invalid")
        seed = self.random_seed
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2 ** 32 - 1:
            raise SandboxSpecError("random seed is invalid")
        for name in ("cpu_quota", "cpu_period", "memory_limit_bytes", "pids_limit", "wall_clock_seconds"):
            if int(getattr(self, name)) <= 0:
                raise SandboxSpecError("%s must be positive" % name)
        if self.uid <= 0 or self.gid <= 0:
            raise SandboxSpecError("sandbox user must not be root")
        if self.output_tmpfs_bytes < 0 or self.tmp_tmpfs_bytes <= 0:
            raise SandboxSpecError("tmpfs bounds are invalid")

    @property
    def argv(self) -> Tuple[str, ...]:
        return tuple(self.entry_command)

    @property
    def cwd(self) -> str:
        return self.working_dir or "/tmp"

    @property
    def socket_dir(self) -> Path:
        return self.socket_path.parent

    @property
    def output_path(self) -> Path:
        return self.output_dir / OUTPUT_FILE_NAME


def sandbox_environment(spec: SandboxSpec) -> Dict[str, str]:
    """The fixed environment supplied by the competition host."""

    environment: Dict[str, str] = {"PATH": PROCESS_ENV["PATH"]}
    environment.update({
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONPATH": SANDBOX_MODEL_DIR,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": str(spec.random_seed),
        "LAB_ARENA_RANDOM_SEED": str(spec.random_seed),
        "LAB_ARENA_EVALUATION_DATE": spec.evaluation_date,
        "LAB_ARENA_INPUT_PATH": SANDBOX_INPUT_PATH,
        "LAB_ARENA_OUTPUT_PATH": SANDBOX_OUTPUT_PATH,
        "LAB_ARENA_WORKER_SOCKET": SANDBOX_SOCKET_PATH,
    })
    environment.update(PROVIDER_BASE_URLS)
    for name, value in spec.extra_environment.items():
        environment.setdefault(name, value)  # the fixed model environment always wins
    return environment


def oci_spec(spec: SandboxSpec) -> Dict[str, Any]:
    """The OCI runtime document for one sandbox (mirrors the enclave shape)."""

    environment = sandbox_environment(spec)
    mounts = [
        {"destination": "/proc", "type": "proc", "source": "proc"},
        {
            "destination": "/dev",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": ["nosuid", "strictatime", "mode=755", "size=65536k"],
        },
        {
            "destination": "/tmp",
            "type": "tmpfs",
            "source": "tmpfs",
            "options": ["nosuid", "nodev", "mode=1777", "size=%d" % spec.tmp_tmpfs_bytes],
        },
        {
            "destination": SANDBOX_INPUT_DIR,
            "type": "bind",
            "source": str(spec.input_dir),
            "options": ["rbind", "ro", "nosuid", "nodev", "noexec"],
        },
        {
            "destination": SANDBOX_OUTPUT_DIR,
            "type": "bind",
            "source": str(spec.output_dir),
            "options": ["rbind", "rw", "nosuid", "nodev", "noexec"],
        },
        {
            "destination": SANDBOX_SOCKET_DIR,
            "type": "bind",
            "source": str(spec.socket_dir),
            "options": ["rbind", "rw", "nosuid", "nodev", "noexec"],
        },
    ]
    if spec.source_dir is not None:
        mounts.extend(
            [
                {
                    "destination": SANDBOX_AGENT_SOURCE_DIR,
                    "type": "bind",
                    "source": str(spec.source_dir),
                    "options": ["rbind", "ro", "nosuid", "nodev"],
                },
                {
                    "destination": SANDBOX_AGENT_DEPENDENCY_DIR,
                    "type": "bind",
                    "source": str(spec.dependency_dir),
                    "options": ["rbind", "ro", "nosuid", "nodev"],
                },
                {
                    "destination": SANDBOX_AGENT_ENTRYPOINT_PATH,
                    "type": "bind",
                    "source": str(spec.agent_entrypoint_path),
                    "options": ["bind", "ro", "nosuid", "nodev", "noexec"],
                },
            ]
        )
    linux = {
        # No "network" namespace: runsc --network=none is the isolation.
        "namespaces": [
            {"type": "pid"},
            {"type": "ipc"},
            {"type": "uts"},
            {"type": "mount"},
            {"type": "user"},
        ],
        # Rootful runsc starts its gofer as namespace root while the model
        # runs as the explicit nobody identity; both must be mapped.
        "uidMappings": [
            {"containerID": 0, "hostID": 0, "size": 1},
            {"containerID": spec.uid, "hostID": spec.uid, "size": 1},
        ],
        "gidMappings": [
            {"containerID": 0, "hostID": 0, "size": 1},
            {"containerID": spec.gid, "hostID": spec.gid, "size": 1},
        ],
        "resources": {
            "memory": {"limit": spec.memory_limit_bytes},
            "cpu": {"quota": spec.cpu_quota, "period": spec.cpu_period},
            "pids": {"limit": spec.pids_limit},
        },
        "maskedPaths": [
            "/dev/log",
            "/dev/nsm",
            "/proc/acpi",
            "/proc/keys",
            "/proc/kcore",
            "/proc/latency_stats",
            "/proc/timer_list",
            "/proc/timer_stats",
            "/sys/firmware",
        ],
        "readonlyPaths": [
            "/proc/asound",
            "/proc/bus",
            "/proc/fs",
            "/proc/irq",
            "/proc/sys",
            "/proc/sysrq-trigger",
        ],
        "seccomp": {
            "defaultAction": "SCMP_ACT_ALLOW",
            "architectures": ["SCMP_ARCH_X86_64"],
            "syscalls": [
                {
                    "names": ["socket"],
                    "action": "SCMP_ACT_ERRNO",
                    "errnoRet": 1,
                    "args": [{"index": 0, "value": LINUX_AF_UNIX, "op": "SCMP_CMP_NE"}],
                },
                {
                    "names": ["mount", "pivot_root", "ptrace", "bpf", "keyctl", "perf_event_open"],
                    "action": "SCMP_ACT_ERRNO",
                    "errnoRet": 1,
                },
            ],
        },
    }
    return {
        "ociVersion": "1.0.2",
        "process": {
            "terminal": False,
            "user": {"uid": spec.uid, "gid": spec.gid},
            "args": list(spec.argv),
            "env": ["%s=%s" % item for item in sorted(environment.items())],
            "cwd": spec.cwd,
            "capabilities": {
                "bounding": [],
                "effective": [],
                "inheritable": [],
                "permitted": [],
                "ambient": [],
            },
            "rlimits": [
                {"type": "RLIMIT_NOFILE", "hard": 1024, "soft": 1024},
                {"type": "RLIMIT_NPROC", "hard": spec.pids_limit, "soft": spec.pids_limit},
            ],
            "noNewPrivileges": True,
        },
        "root": {"path": str(spec.rootfs_path), "readonly": True},
        "hostname": SANDBOX_HOSTNAME,
        "mounts": mounts,
        "linux": linux,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def runsc_run_command(config: RuntimeConfig, runsc_root: Path, bundle: Path, sandbox_id: str) -> List[str]:
    return [
        str(config.runsc_path),
        "--root=%s" % runsc_root,
        # Rootful: the OCI document maps the explicit 65534 identity; gVisor
        # rootless mode would remap the caller to container root instead.
        "--rootless=false",
        "--network=none",
        # Only bind-mounted host Unix sockets are reachable, and only to open.
        "--host-uds=open",
        "--platform=%s" % config.platform,
        "run",
        "--bundle=%s" % bundle,
        sandbox_id,
    ]


def runsc_kill_command(config: RuntimeConfig, runsc_root: Path, sandbox_id: str) -> List[str]:
    return [str(config.runsc_path), "--root=%s" % runsc_root, "kill", sandbox_id, "KILL"]


def runsc_delete_command(config: RuntimeConfig, runsc_root: Path, sandbox_id: str) -> List[str]:
    return [str(config.runsc_path), "--root=%s" % runsc_root, "delete", "--force", sandbox_id]


def output_mount_command(spec: SandboxSpec) -> List[str]:
    """Size-bound host tmpfs behind ``/output`` (requires root)."""

    options = "size=%d,mode=0700,uid=%d,gid=%d,nosuid,nodev,noexec" % (spec.output_tmpfs_bytes, spec.uid, spec.gid)
    return ["mount", "-t", "tmpfs", "-o", options, "tmpfs", str(spec.output_dir)]


def output_unmount_command(spec: SandboxSpec) -> List[str]:
    return ["umount", str(spec.output_dir)]


# ---------------------------------------------------------------------------
# Results and output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxResult:
    exit_code: Optional[int]
    timed_out: bool
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    wall_seconds: float
    cpu_seconds: float
    max_rss_bytes: int
    output_bytes: Optional[bytes]
    output_path: str
    output_error: Optional[str] = None


def read_output(spec: SandboxSpec, *, max_bytes: int = MAX_OUTPUT_BYTES) -> Optional[bytes]:
    """Read ``companies.json``: ``None`` when absent, error when unbounded.

    Symlinks, directories, and files above the cap fail closed; the size is
    enforced while reading, not from a stat the model could race.
    """

    path = spec.output_path
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SandboxOutputError("output is unreadable") from exc
    if not os.path.isfile(path) or os.path.islink(path) or info.st_size > max_bytes:
        raise SandboxOutputError("output is not a bounded regular file")
    try:
        with open(path, "rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise SandboxOutputError("output is unreadable") from exc
    if len(data) > max_bytes:
        raise SandboxOutputError("output exceeds %d bytes" % max_bytes)
    return data


class _BoundedCapture:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.chunks: List[bytes] = []
        self.captured = 0
        self.truncated = False
        self.failed = False

    def append(self, chunk: bytes) -> None:
        remaining = self.limit - self.captured
        if remaining > 0:
            kept = chunk[:remaining]
            self.chunks.append(kept)
            self.captured += len(kept)
        if len(chunk) > max(0, remaining):
            self.truncated = True

    def value(self) -> bytes:
        return b"".join(self.chunks)


def _drain(stream: Any, capture: _BoundedCapture) -> None:
    try:
        while True:
            chunk = stream.read(PIPE_CHUNK_BYTES)
            if not chunk:
                break
            capture.append(chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk))
    except (OSError, ValueError):
        capture.failed = True
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _children_rusage() -> Tuple[float, int]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss = int(usage.ru_maxrss) if sys.platform == "darwin" else int(usage.ru_maxrss) * 1024
    return float(usage.ru_utime + usage.ru_stime), rss


def _run_command(process_runner: ProcessRunner, argv: Sequence[str], *, timeout: float) -> int:
    process = process_runner(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(PROCESS_ENV),
        start_new_session=True,
    )
    try:
        process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise ArenaRuntimeError("command timed out: %s" % argv[-2:]) from exc
    return int(process.returncode)


def _stop_process(process: Any, *, grace: float) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired as exc:
        raise ArenaRuntimeError("sandbox launcher could not be stopped") from exc


def _prepare_output_dir(spec: SandboxSpec) -> None:
    if spec.output_dir.exists():
        if not spec.output_dir.is_dir() or spec.output_dir.is_symlink() or any(spec.output_dir.iterdir()):
            raise SandboxSpecError("output directory must be empty")
    else:
        spec.output_dir.mkdir(parents=True, mode=0o700)


def prepare_sandbox_access(
    spec: SandboxSpec,
    *,
    chown: Callable[[Any, int, int], None] = os.chown,
) -> None:
    """Give only the sandbox user access to its input and worker socket.

    The runner is root in production, while ``tempfile.mkdtemp`` and a secure
    umask create root-only paths. The paths are bind-mounted directly into
    the sandbox, so their owner and mode must be set before runsc starts.
    """

    input_file = spec.input_dir / INPUT_FILE_NAME
    try:
        input_dir_info = os.lstat(spec.input_dir)
        input_info = os.lstat(input_file)
        socket_dir_info = os.lstat(spec.socket_dir)
        socket_info = os.lstat(spec.socket_path)
    except OSError as exc:
        raise SandboxSpecError("sandbox input or worker socket is unavailable") from exc
    if (
        not stat.S_ISDIR(input_dir_info.st_mode)
        or stat.S_ISLNK(input_dir_info.st_mode)
        or not stat.S_ISREG(input_info.st_mode)
        or stat.S_ISLNK(input_info.st_mode)
        or not stat.S_ISDIR(socket_dir_info.st_mode)
        or stat.S_ISLNK(socket_dir_info.st_mode)
        or not stat.S_ISSOCK(socket_info.st_mode)
    ):
        raise SandboxSpecError("sandbox input or worker socket has an unsafe type")
    for path in (spec.input_dir, input_file, spec.socket_dir, spec.socket_path):
        chown(path, spec.uid, spec.gid)
    os.chmod(spec.input_dir, 0o500)
    os.chmod(input_file, 0o400)
    os.chmod(spec.socket_dir, 0o700)
    os.chmod(spec.socket_path, 0o600)


def require_safe_agent_mounts(spec: SandboxSpec) -> None:
    """Check the host-owned source mounts immediately before runsc starts."""

    if spec.source_dir is None:
        return
    try:
        source_info = os.lstat(spec.source_dir)
        dependency_info = os.lstat(spec.dependency_dir)
        entrypoint_info = os.lstat(spec.agent_entrypoint_path)
    except OSError as exc:
        raise SandboxSpecError("agent source mount is unavailable") from exc
    if (
        not stat.S_ISDIR(source_info.st_mode)
        or stat.S_ISLNK(source_info.st_mode)
        or not stat.S_ISDIR(dependency_info.st_mode)
        or stat.S_ISLNK(dependency_info.st_mode)
        or not stat.S_ISREG(entrypoint_info.st_mode)
        or stat.S_ISLNK(entrypoint_info.st_mode)
    ):
        raise SandboxSpecError("agent source mount has an unsafe type")


def run_sandbox(
    config: RuntimeConfig,
    spec: SandboxSpec,
    *,
    process_runner: ProcessRunner = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    rusage: Callable[[], Tuple[float, int]] = _children_rusage,
) -> SandboxResult:
    """Run one sandbox to completion or timeout and always clean up.

    Writes the bundle, mounts the bounded output tmpfs, runs runsc with a hard
    deadline of ``wall_clock_seconds + TIMEOUT_GRACE_SECONDS``, kills on
    timeout, reads ``companies.json`` into the result, then deletes the
    sandbox, unmounts, and removes the output and bundle directories. Cleanup
    runs every step even after a failure and raises ``SandboxCleanupError``
    if any step failed.
    """

    if not config.work_dir.is_dir():
        raise SandboxSpecError("runtime work directory is unavailable")
    if not spec.input_dir.is_dir() or not (spec.input_dir / INPUT_FILE_NAME).is_file():
        raise SandboxSpecError("input directory must contain %s" % INPUT_FILE_NAME)
    require_safe_agent_mounts(spec)
    bundle = Path(tempfile.mkdtemp(prefix="lab-arena-%s-" % spec.sandbox_id, dir=config.work_dir))
    runsc_root = bundle / "runsc"
    cleanup_errors: List[str] = []
    output_mounted = False
    process: Any = None
    threads: List[threading.Thread] = []
    result: Optional[SandboxResult] = None
    try:
        runsc_root.mkdir(mode=0o700)
        config_path = bundle / "config.json"
        config_path.write_text(json.dumps(oci_spec(spec), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(config_path, 0o600)
        _prepare_output_dir(spec)
        if spec.output_tmpfs_bytes > 0:
            if _run_command(process_runner, output_mount_command(spec), timeout=config.cleanup_timeout_seconds) != 0:
                raise ArenaRuntimeError("output tmpfs mount failed")
            output_mounted = True
        else:
            os.chown(spec.output_dir, spec.uid, spec.gid)

        stdout_capture = _BoundedCapture(MAX_LOG_BYTES)
        stderr_capture = _BoundedCapture(MAX_LOG_BYTES)
        cpu_before, _ = rusage()
        started = clock()
        process = process_runner(
            runsc_run_command(config, runsc_root, bundle, spec.sandbox_id),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(PROCESS_ENV),
            cwd=str(bundle),
            start_new_session=True,
        )
        for stream, capture in ((process.stdout, stdout_capture), (process.stderr, stderr_capture)):
            thread = threading.Thread(target=_drain, args=(stream, capture), daemon=True)
            thread.start()
            threads.append(thread)
        deadline = started + float(spec.wall_clock_seconds) + TIMEOUT_GRACE_SECONDS
        timed_out = False
        while process.poll() is None:
            if clock() >= deadline:
                timed_out = True
                break
            sleep(0.05)
        if timed_out:
            try:
                _run_command(process_runner, runsc_kill_command(config, runsc_root, spec.sandbox_id), timeout=config.cleanup_timeout_seconds)
            except ArenaRuntimeError as exc:
                cleanup_errors.append("kill: %s" % exc)
            _stop_process(process, grace=STOP_GRACE_SECONDS)
        wall_seconds = max(0.0, clock() - started)
        for thread in threads:
            thread.join(STOP_GRACE_SECONDS)
        if any(thread.is_alive() for thread in threads):
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
            for thread in threads:
                thread.join(STOP_GRACE_SECONDS)
        if any(thread.is_alive() for thread in threads):
            raise ArenaRuntimeError("sandbox pipes did not close")
        if stdout_capture.failed or stderr_capture.failed:
            raise ArenaRuntimeError("sandbox pipe read failed")
        cpu_after, max_rss = rusage()
        output_bytes: Optional[bytes] = None
        output_error: Optional[str] = None
        try:
            output_bytes = read_output(spec)
        except SandboxOutputError as exc:
            output_error = str(exc)
        result = SandboxResult(
            exit_code=None if timed_out else int(process.returncode),
            timed_out=timed_out,
            stdout=stdout_capture.value(),
            stderr=stderr_capture.value(),
            stdout_truncated=stdout_capture.truncated,
            stderr_truncated=stderr_capture.truncated,
            wall_seconds=wall_seconds,
            cpu_seconds=max(0.0, cpu_after - cpu_before),
            max_rss_bytes=max_rss,
            output_bytes=output_bytes,
            output_path=str(spec.output_path),
            output_error=output_error,
        )
    finally:
        if process is not None and process.poll() is None:
            try:
                _stop_process(process, grace=STOP_GRACE_SECONDS)
            except ArenaRuntimeError as exc:
                cleanup_errors.append("launcher: %s" % exc)
        try:
            if _run_command(process_runner, runsc_delete_command(config, runsc_root, spec.sandbox_id), timeout=config.cleanup_timeout_seconds) != 0:
                cleanup_errors.append("delete: nonzero exit")
        except (ArenaRuntimeError, OSError) as exc:
            cleanup_errors.append("delete: %s" % type(exc).__name__)
        if output_mounted:
            try:
                if _run_command(process_runner, output_unmount_command(spec), timeout=config.cleanup_timeout_seconds) != 0:
                    cleanup_errors.append("umount: nonzero exit")
            except (ArenaRuntimeError, OSError) as exc:
                cleanup_errors.append("umount: %s" % type(exc).__name__)
        for directory in (spec.output_dir, bundle):
            try:
                shutil.rmtree(directory)
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_errors.append("rmtree %s: %s" % (directory.name, type(exc).__name__))
        if cleanup_errors:
            raise SandboxCleanupError("sandbox cleanup incomplete: " + "; ".join(cleanup_errors))
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Runtime front doors
# ---------------------------------------------------------------------------


class RunscRuntime:
    """Host-checked runtime: constructing it proves Linux x86_64 and the binary."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        process_runner: ProcessRunner = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        require_linux_x86_64()
        self.config = config
        require_runsc_executable(config.runsc_path)
        self._process_runner = process_runner
        self._clock = clock
        self._sleep = sleep

    def run_icp(
        self,
        spec: SandboxSpec,
        *,
        process_runner: Optional[ProcessRunner] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> SandboxResult:
        prepare_sandbox_access(spec)
        return run_sandbox(
            self.config,
            spec,
            process_runner=process_runner or self._process_runner,
            clock=clock or self._clock,
            sleep=self._sleep,
        )

    def read_output(self, spec: SandboxSpec) -> Optional[bytes]:
        return read_output(spec)


class FakeRuntime:
    """Test double: records specs and returns preset results in order."""

    def __init__(self, results: Sequence[SandboxResult] = ()) -> None:
        self.specs: List[SandboxSpec] = []
        self.results: List[SandboxResult] = list(results)

    def run_icp(self, spec: SandboxSpec, **_: Any) -> SandboxResult:
        self.specs.append(spec)
        if not self.results:
            raise ArenaRuntimeError("fake runtime has no result for %s" % spec.sandbox_id)
        return self.results.pop(0)

    def read_output(self, spec: SandboxSpec) -> Optional[bytes]:
        return read_output(spec)


def fake_result(
    *,
    exit_code: Optional[int] = 0,
    timed_out: bool = False,
    output_bytes: Optional[bytes] = None,
    stdout: bytes = b"",
    stderr: bytes = b"",
    output_error: Optional[str] = None,
) -> SandboxResult:
    """Convenience constructor for ``FakeRuntime`` results."""

    return SandboxResult(
        exit_code=exit_code,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=False,
        stderr_truncated=False,
        wall_seconds=1.0,
        cpu_seconds=0.5,
        max_rss_bytes=1024 * 1024,
        output_bytes=output_bytes,
        output_path=SANDBOX_OUTPUT_PATH,
        output_error=output_error,
    )


__all__ = [
    "ArenaRuntimeError",
    "AGENT_ENTRY_COMMAND",
    "AGENT_WORKING_DIR",
    "DEFAULT_PLATFORM",
    "FakeRuntime",
    "INPUT_FILE_NAME",
    "MAX_LOG_BYTES",
    "MAX_OUTPUT_BYTES",
    "OUTPUT_FILE_NAME",
    "OUTPUT_TMPFS_BYTES",
    "PROVIDER_BASE_URLS",
    "RunscRuntime",
    "RuntimeConfig",
    "RuntimeHostError",
    "SANDBOX_INPUT_DIR",
    "SANDBOX_OUTPUT_DIR",
    "SANDBOX_SOCKET_DIR",
    "SANDBOX_SOCKET_PATH",
    "SandboxCleanupError",
    "SandboxOutputError",
    "SandboxResult",
    "SandboxSpec",
    "SandboxSpecError",
    "SCORER_ENTRY_COMMAND",
    "SCORER_WORKING_DIR",
    "TIMEOUT_GRACE_SECONDS",
    "fake_result",
    "oci_spec",
    "output_mount_command",
    "output_unmount_command",
    "prepare_sandbox_access",
    "read_output",
    "require_linux_x86_64",
    "run_sandbox",
    "runsc_delete_command",
    "runsc_kill_command",
    "runsc_run_command",
    "sandbox_environment",
    "require_runsc_executable",
]
