#!/usr/bin/env python3
"""Run the exact local N-1 restart and V2 publication rehearsal.

The driver has two deliberately fixed profiles:

``prepush``
    The default 5-10 minute gate: one forward N-1 -> N restart and one
    complete V2 publication in a resource-bounded Docker replica.
``unaccelerated``
    Forward, rollback, roll-forward, the external-boundary fault matrix,
    concurrency checks, and 100 accelerated stateful subnet epochs.

Neither profile reads production credentials.  Repository-owned behavior is
executed from the frozen candidate checkout; only the boundaries enumerated by
the rehearsal contract may be implemented locally.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import errno
import fcntl
from functools import partial
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Optional, Sequence
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_REPOSITORY = "leadpoet-local-restart-rehearsal"
_LOCAL_IMAGE_OPERATION_TIMEOUT_SECONDS = 10.0
_REHEARSAL_BASE_PULL_TIMEOUT_SECONDS = 120.0
_BUILDX_OPERATION_TIMEOUT_SECONDS = 10.0
_DRAND_SOURCE_DOWNLOAD_TIMEOUT_SECONDS = 120.0
_DRAND_SOURCE_DOWNLOAD_PROCESS_TIMEOUT_SECONDS = 125.0
_DRAND_BUILDER_TIMEOUT_SECONDS = 180.0
_DRAND_COMPILE_TIMEOUT_SECONDS = 300.0
_DRAND_SOURCE_MAX_BYTES = 64 * 1024 * 1024
_DRAND_INTERNAL_DOWNLOAD_COMMAND = "--internal-download-drand-source"
_GIT_BLOB_READ_TIMEOUT_SECONDS = 30.0
_SYSTEM_BUILDX_PATHS = (
    Path("/usr/local/lib/docker/cli-plugins/docker-buildx"),
    Path("/usr/local/libexec/docker/cli-plugins/docker-buildx"),
    Path("/usr/lib/docker/cli-plugins/docker-buildx"),
    Path("/usr/libexec/docker/cli-plugins/docker-buildx"),
)
REHEARSAL_LOCK_PATH = (
    Path.home()
    / ".cache"
    / "leadpoet-local-restart-rehearsal"
    / "controller.lock"
)
REHEARSAL_BASE_IMAGES = {
    "linux/amd64": (
        "public.ecr.aws/amazonlinux/amazonlinux@sha256:"
        "7dfb72e165c7b2f5fd2ee050c202160ee0cced24991f14736b831221f2004eee"
    ),
    "linux/arm64": (
        "public.ecr.aws/amazonlinux/amazonlinux@sha256:"
        "d23b77c815875a32165bc160248a6fcaf932dbbcdb7adc157680c39e4d254b38"
    ),
}
PYTHON37_IMAGE = (
    "python@sha256:"
    "b53f496ca43e5af6994f8e316cf03af31050bf7944e0e4a308ad86c001cf028b"
)
COMMITTED_HARNESS_PATHS = (
    "tests/restart_rehearsal/Dockerfile",
    "tests/restart_rehearsal/artifact_identity.py",
    "tests/restart_rehearsal/boundary_contract.json",
    "tests/restart_rehearsal/compact_weight_joined_runner.py",
    "tests/restart_rehearsal/contract_adapter.py",
    "tests/restart_rehearsal/fixture_contract.py",
    "tests/restart_rehearsal/fixtures/production_shaped_v2.json",
    "tests/restart_rehearsal/fixtures/subtensor_metadata_spec452_parent8984915.scale.gz",
    "tests/restart_rehearsal/gateway_boundary_service.py",
    "tests/restart_rehearsal/gateway_enclave_service.py",
    "tests/restart_rehearsal/join_evidence.py",
    "tests/restart_rehearsal/local_services.py",
    "tests/restart_rehearsal/prepare_external_artifacts.py",
    "tests/restart_rehearsal/prepare_host_fixtures.py",
    "tests/restart_rehearsal/prepare_scoring_wheelhouse_aliases.py",
    "tests/restart_rehearsal/postgres_v2_contract_probe.py",
    "tests/restart_rehearsal/production_workflow_runner.py",
    "tests/restart_rehearsal/sanitized_weight_fixture.py",
    "tests/restart_rehearsal/sitecustomize.py",
    "tests/restart_rehearsal/tls_connect_proxy_service.py",
    "tests/restart_rehearsal/validator_enclave_service.py",
    "tests/restart_rehearsal/weight_readiness_runner.py",
    "tests/restart_rehearsal/run_inside.sh",
    "tests/restart_rehearsal/verify_evidence.py",
)
SCORING_WHEELHOUSE_PATHS = (
    "gateway/tee/requirements-scoring-py39.in",
    "gateway/tee/requirements-scoring-py39.lock",
)
EXTERNAL_ARTIFACT_LOCK_PATHS = (
    "gateway/tee/runsc-runtime.lock.json",
    "validator_tee/runtime-artifacts-v2.lock.json",
)
PROFILE_LIMITS = {
    "prepush": {
        "cpus": "4",
        "memory": "7g",
        "epochs": 1,
        "fault_matrix": False,
        "target_seconds": 600,
    },
    "release": {
        "cpus": "6",
        "memory": "7g",
        "epochs": 100,
        "fault_matrix": True,
        "target_seconds": None,
    },
}
CLI_PROFILES = ("prepush", "unaccelerated")


class RehearsalTimeBudgetExceeded(TimeoutError):
    """The bounded prepush rehearsal exceeded its wall-clock budget."""


def _runtime_profile(cli_profile: str) -> str:
    if cli_profile == "unaccelerated":
        return "release"
    if cli_profile == "prepush":
        return cli_profile
    raise ValueError(f"unsupported rehearsal profile: {cli_profile}")


@contextmanager
def _profile_time_limit(seconds: Optional[int]) -> Iterator[None]:
    if seconds is None:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise RehearsalTimeBudgetExceeded(
            f"prepush rehearsal exceeded its {seconds}-second wall-clock budget"
        )

    signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    started = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, previous_delay - elapsed),
                previous_interval,
            )


@contextmanager
def _exclusive_rehearsal_lock() -> Iterator[None]:
    REHEARSAL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REHEARSAL_LOCK_PATH.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "owner metadata unavailable"
            raise SystemExit(
                "another exact local restart rehearsal is already running; "
                f"lock={REHEARSAL_LOCK_PATH} owner={owner}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "cwd": str(Path.cwd()),
                    "pid": os.getpid(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        handle.flush()
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _isolated_docker_client_config() -> Iterator[Path]:
    """Keep Docker credentials and Buildx state private to this rehearsal."""

    def managed_environment_key(name: str) -> bool:
        return (
            name in {
                "DOCKER_AUTH_CONFIG",
                "DOCKER_BUILDKIT",
                "DOCKER_CONFIG",
                "EXPERIMENTAL_BUILDKIT_SOURCE_POLICY",
            }
            or name.startswith("DOCKER_CLI_PLUGIN_")
            or name.startswith("BUILDX_")
            or name.startswith("BUILDKIT_")
        )

    previous = {
        name: value
        for name, value in os.environ.items()
        if managed_environment_key(name)
    }
    with tempfile.TemporaryDirectory(
        prefix="leadpoet-restart-docker-config-"
    ) as raw:
        root = Path(raw)
        root.chmod(0o700)
        config = root / "config.json"
        config.write_text('{"auths":{}}\n', encoding="utf-8")
        config.chmod(0o600)
        buildx_state = root / "buildx"
        buildx_state.mkdir(mode=0o700)
        for name in tuple(os.environ):
            if managed_environment_key(name):
                os.environ.pop(name, None)
        os.environ["DOCKER_CONFIG"] = str(root)
        os.environ["DOCKER_BUILDKIT"] = "1"
        os.environ["BUILDX_CONFIG"] = str(buildx_state)
        try:
            yield root
        finally:
            for name in tuple(os.environ):
                if managed_environment_key(name):
                    os.environ.pop(name, None)
            os.environ.update(previous)


def _buildx_candidate_paths(docker_executable: Path) -> tuple[Path, ...]:
    """Return only the exact conventional buildx locations for this Docker."""

    prefix = docker_executable.parent.parent
    paths = (
        prefix / "cli-plugins" / "docker-buildx",
        prefix / "lib" / "docker" / "cli-plugins" / "docker-buildx",
        prefix / "libexec" / "docker" / "cli-plugins" / "docker-buildx",
        *_SYSTEM_BUILDX_PATHS,
    )
    return tuple(dict.fromkeys(paths))


def _resolve_official_buildx_executable() -> Path:
    """Resolve one non-writable executable buildx from the Docker install."""

    docker_command = shutil.which("docker")
    if docker_command is None:
        raise SystemExit("Docker CLI is unavailable for the restart rehearsal")
    try:
        docker_executable = Path(docker_command).resolve(strict=True)
    except OSError as exc:
        raise SystemExit("Docker CLI path is invalid") from exc
    docker_metadata = docker_executable.stat()
    if (
        not docker_executable.is_file()
        or not os.access(docker_executable, os.X_OK)
        or docker_metadata.st_uid not in {0, os.getuid()}
        or docker_metadata.st_mode & 0o022
    ):
        raise SystemExit("Docker CLI is not a trusted executable file")

    resolved_by_inode: dict[tuple[int, int], Path] = {}
    prefix = docker_executable.parent.parent
    for candidate in _buildx_candidate_paths(docker_executable):
        if not os.path.lexists(candidate):
            continue
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.stat()
        except OSError as exc:
            raise SystemExit("installed Docker buildx path is invalid") from exc
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in resolved_by_inode:
            continue
        try:
            candidate.relative_to(prefix)
            resolved.relative_to(prefix)
            coinstalled = True
        except ValueError:
            coinstalled = False
        if (
            not resolved.is_file()
            or not os.access(resolved, os.X_OK)
            or metadata.st_mode & 0o022
            or (
                coinstalled
                and metadata.st_uid != docker_metadata.st_uid
            )
            or (
                not coinstalled
                and (
                    candidate not in _SYSTEM_BUILDX_PATHS
                    or metadata.st_uid != 0
                )
            )
        ):
            raise SystemExit("installed Docker buildx is not a trusted executable")
        resolved_by_inode[inode] = resolved

    if len(resolved_by_inode) != 1:
        raise SystemExit(
            "restart rehearsal requires exactly one installed Docker buildx"
        )
    return next(iter(resolved_by_inode.values()))


def _provision_official_buildx(docker_client_root: Path) -> Path:
    """Stage and validate exactly one official buildx inside private state."""

    target = _resolve_official_buildx_executable()
    bin_root = docker_client_root / "bin"
    try:
        bin_root.mkdir(mode=0o700)
        staged = bin_root / "docker-buildx"
        staged.symlink_to(target)
        if staged.resolve(strict=True) != target:
            raise RuntimeError("staged buildx target differs")
    except OSError as exc:
        raise SystemExit("unable to stage Docker buildx privately") from exc

    try:
        result = _run(
            [str(staged), "docker-cli-plugin-metadata"],
            capture=True,
            timeout_seconds=_BUILDX_OPERATION_TIMEOUT_SECONDS,
        )
    except (KeyboardInterrupt, RehearsalTimeBudgetExceeded):
        raise
    except (subprocess.SubprocessError, OSError) as exc:
        raise SystemExit("unable to validate installed Docker buildx") from exc
    try:
        metadata = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SystemExit("installed Docker buildx metadata is malformed") from exc
    version = metadata.get("Version") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("SchemaVersion") != "0.1.0"
        or metadata.get("Vendor") != "Docker Inc."
        or metadata.get("ShortDescription") != "Docker Buildx"
        or not isinstance(version, str)
        or re.fullmatch(
            r"v?[0-9]+\.[0-9]+\.[0-9]+(?:[-+._][0-9A-Za-z.-]+)*",
            version,
        )
        is None
    ):
        raise SystemExit("installed Docker buildx metadata is untrusted")
    return staged


_WORKER_PROCESS_STATE = threading.local()
_WORKER_TERMINATE_GRACE_SECONDS = 2.0
_WORKER_DOCKER_CLEANUP_SECONDS = 2.0
_WORKER_DOCKER_REMOVAL_SECONDS = 12.0
_WORKER_DOCKER_CONVERGENCE_SECONDS = 6.0
_WORKER_DOCKER_ABSENCE_OBSERVATIONS = 3
_WORKER_DOCKER_POLL_SECONDS = 0.05
_EVIDENCE_NORMALIZATION_TIMEOUT_SECONDS = 30.0
_EVIDENCE_NORMALIZATION_MAX_PASSES = 32
_WORKFLOW_FAILURE_PROJECTION_MAX_BYTES = 16 * 1024 * 1024
_WORKFLOW_FAILURE_PROJECTION_MAX_STAGES = 16
_SAFE_WORKFLOW_PROJECTION_ERROR_TYPES = frozenset(
    {
        "AssertionError",
        "CalledProcessError",
        "FileNotFoundError",
        "OSError",
        "PermissionError",
        "RuntimeError",
        "SystemExit",
        "TimeoutError",
        "TypeError",
        "ValueError",
    }
)
_SCHEDULER_WORKER_READY_SECONDS = 2.0
_DEFERRED_PROCESS_SIGNALS = frozenset((signal.SIGALRM, signal.SIGINT))
_SOURCE_GIT_SAFE_DIRECTORY_DOCKER_ARGS = (
    "--env",
    "GIT_CONFIG_COUNT=1",
    "--env",
    "GIT_CONFIG_KEY_0=safe.directory",
    "--env",
    "GIT_CONFIG_VALUE_0=/source",
)


class _EvidenceNormalizationError(RuntimeError):
    """Safe categorical failure for the host evidence handoff."""

    def __init__(self, *, phase: str, category: str, status: int) -> None:
        self.phase = phase
        self.category = category
        self.status = status
        self.diagnostics = [
            {"category": category, "phase": phase, "status": status}
        ]
        super().__init__(
            "evidence normalization failed "
            f"phase={phase} category={category} status={status}"
        )


def _terminate_worker_process(process: subprocess.Popen[Any]) -> None:
    """Terminate, escalate, and reap one exact worker-owned subprocess."""

    if process.poll() is not None:
        process.wait(timeout=0)
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_WORKER_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=_WORKER_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"worker subprocess pid={process.pid} could not be reaped"
            ) from exc


def _remove_worker_container(container_name: str) -> None:
    """Force-remove one run-owned container until absence is stably proven."""

    started_at = time.monotonic()
    removal_deadline = started_at + _WORKER_DOCKER_REMOVAL_SECONDS
    hard_deadline = removal_deadline + _WORKER_DOCKER_CONVERGENCE_SECONDS
    absence_started_at: float | None = None
    consecutive_absent_observations = 0
    while True:
        now = time.monotonic()
        if now > hard_deadline:
            break
        command_timeout = max(
            0.001,
            min(_WORKER_DOCKER_CLEANUP_SECONDS, hard_deadline - now),
        )
        try:
            subprocess.run(
                ["docker", "container", "rm", "--force", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=command_timeout,
            )
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        if now > hard_deadline:
            break
        command_timeout = max(
            0.001,
            min(_WORKER_DOCKER_CLEANUP_SECONDS, hard_deadline - now),
        )
        try:
            inspected = subprocess.run(
                ["docker", "container", "inspect", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=command_timeout,
            )
        except subprocess.TimeoutExpired:
            absent = False
        else:
            error = (inspected.stderr or "").lower()
            absent = (
                inspected.returncode != 0
                and (
                    "no such container" in error
                    or "no such object" in error
                )
            )
        observed_at = time.monotonic()
        if absent:
            if absence_started_at is None:
                absence_started_at = observed_at
                consecutive_absent_observations = 1
            else:
                consecutive_absent_observations += 1
            if (
                consecutive_absent_observations
                >= _WORKER_DOCKER_ABSENCE_OBSERVATIONS
                and observed_at - absence_started_at
                >= _WORKER_DOCKER_CONVERGENCE_SECONDS
            ):
                return
        else:
            absence_started_at = None
            consecutive_absent_observations = 0
        if observed_at >= hard_deadline:
            break
        remaining = hard_deadline - observed_at
        if remaining > 0:
            time.sleep(min(_WORKER_DOCKER_POLL_SECONDS, remaining))
    raise RuntimeError(
        "worker container removal and stable absence did not converge within "
        f"{_WORKER_DOCKER_REMOVAL_SECONDS:g}s removal plus "
        f"{_WORKER_DOCKER_CONVERGENCE_SECONDS:g}s proof "
        f"name={container_name}"
    )


def _annotate_worker_cleanup_errors(
    exc: BaseException,
    errors: Sequence[str],
) -> None:
    if not errors:
        return
    message = "worker cleanup failed: " + "; ".join(errors)
    try:
        print(
            f"REHEARSAL_WORKER_CLEANUP_FAILED error={message!r}",
            file=sys.stderr,
            flush=True,
        )
    except BaseException:
        pass
    add_note = getattr(exc, "add_note", None)
    if callable(add_note):
        try:
            add_note(message)
        except BaseException:
            pass


class _WorkerProcessRegistry:
    """Own subprocesses launched by one cancellable prepush worker."""

    def __init__(self) -> None:
        self._cancelled = False
        self._cleanup_results: dict[
            subprocess.Popen[Any], tuple[str, ...]
        ] = {}
        self._container_ordinal = 0
        self._lock = threading.Lock()
        self._processes: dict[subprocess.Popen[Any], str | None] = {}
        self._run_id = f"{os.getpid()}-{time.monotonic_ns():x}"
        self._termination_lock = threading.Lock()

    def ensure_accepting(self) -> None:
        with self._lock:
            cancelled = self._cancelled
        if cancelled:
            raise RehearsalTimeBudgetExceeded(
                "prepush worker cancelled at the total rehearsal deadline"
            )

    def prepare_command(self, args: Sequence[str]) -> tuple[list[str], str | None]:
        command = list(args)
        with self._lock:
            cancelled = self._cancelled
            container_name = None
            if not cancelled and command[:2] == ["docker", "run"]:
                if any(
                    item == "--name" or item.startswith("--name=")
                    for item in command[2:]
                ):
                    raise ValueError("worker docker run already declares --name")
                self._container_ordinal += 1
                container_name = (
                    f"leadpoet-prepush-worker-{self._run_id}-"
                    f"{self._container_ordinal}"
                )
        if cancelled:
            self.ensure_accepting()
        if container_name is not None:
            command[2:2] = ["--name", container_name]
        return command, container_name

    def register(
        self,
        process: subprocess.Popen[Any],
        container_name: str | None,
    ) -> None:
        with self._lock:
            cancelled = self._cancelled
            if not cancelled:
                self._processes[process] = container_name
        if cancelled:
            errors = self.terminate(process, container_name)
            try:
                self.ensure_accepting()
            except RehearsalTimeBudgetExceeded as exc:
                _annotate_worker_cleanup_errors(exc, errors)
                raise

    def unregister(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes.pop(process, None)

    def terminate(
        self,
        process: subprocess.Popen[Any],
        container_name: str | None,
    ) -> tuple[str, ...]:
        with self._termination_lock:
            cached = self._cleanup_results.get(process)
            if cached is not None:
                return cached
            errors = []
            try:
                _terminate_worker_process(process)
            except BaseException as exc:
                errors.append(f"subprocess:{type(exc).__name__}:{exc}")
            if container_name is not None:
                try:
                    _remove_worker_container(container_name)
                except BaseException as exc:
                    errors.append(f"container:{type(exc).__name__}:{exc}")
            try:
                if process.poll() is None:
                    _terminate_worker_process(process)
            except BaseException as exc:
                errors.append(f"reap:{type(exc).__name__}:{exc}")
            result = tuple(errors)
            self._cleanup_results[process] = result
            return result

    def cancel(self) -> tuple[str, ...]:
        with self._lock:
            self._cancelled = True
            processes = tuple(self._processes.items())
        errors = []
        for process, container_name in processes:
            errors.extend(self.terminate(process, container_name))
        return tuple(errors)


@contextmanager
def _worker_process_scope(
    registry: _WorkerProcessRegistry,
) -> Iterator[None]:
    previous = getattr(_WORKER_PROCESS_STATE, "registry", None)
    _WORKER_PROCESS_STATE.registry = registry
    try:
        yield
    finally:
        if previous is None:
            del _WORKER_PROCESS_STATE.registry
        else:
            _WORKER_PROCESS_STATE.registry = previous


@contextmanager
def _defer_main_spawn_signals() -> Iterator[None]:
    """Deliver deadline/interrupt signals only after process registration."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.pthread_sigmask(
        signal.SIG_BLOCK,
        _DEFERRED_PROCESS_SIGNALS,
    )
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


@contextmanager
def _signal_masked_worker_executor() -> Iterator[ThreadPoolExecutor]:
    """Create one worker with ALRM/INT blocked for its whole lifetime."""

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("signal-masked worker executor requires main thread")
    ready = threading.Event()

    def initialize_worker() -> None:
        signal.pthread_sigmask(signal.SIG_BLOCK, _DEFERRED_PROCESS_SIGNALS)
        ready.set()

    executor: ThreadPoolExecutor | None = None
    try:
        previous = signal.pthread_sigmask(
            signal.SIG_BLOCK,
            _DEFERRED_PROCESS_SIGNALS,
        )
        try:
            executor = ThreadPoolExecutor(
                max_workers=1,
                initializer=initialize_worker,
            )
            bootstrap = executor.submit(lambda: None)
            if not ready.wait(timeout=_SCHEDULER_WORKER_READY_SECONDS):
                raise RuntimeError("signal-masked worker failed readiness")
            bootstrap.result(timeout=_SCHEDULER_WORKER_READY_SECONDS)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        yield executor
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def _spawn_registered_process(
    command: Sequence[str],
    *,
    registry: _WorkerProcessRegistry | None,
    container_name: str | None,
    **kwargs: Any,
) -> subprocess.Popen[Any]:
    process: subprocess.Popen[Any] | None = None
    registered = False
    try:
        with _defer_main_spawn_signals():
            process = subprocess.Popen(list(command), **kwargs)
            if registry is not None:
                registry.register(process, container_name)
                registered = True
        return process
    except BaseException as exc:
        errors = []
        if process is not None:
            try:
                if registry is None:
                    _terminate_worker_process(process)
                else:
                    errors.extend(registry.terminate(process, container_name))
            except BaseException as cleanup_exc:
                errors.append(
                    f"subprocess:{type(cleanup_exc).__name__}:{cleanup_exc}"
                )
            if registered and registry is not None:
                registry.unregister(process)
        _annotate_worker_cleanup_errors(exc, errors)
        raise


def _run(
    args: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    capture: bool = False,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    registry = getattr(_WORKER_PROCESS_STATE, "registry", None)
    if registry is not None:
        command, container_name = registry.prepare_command(args)
        process = _spawn_registered_process(
            command,
            registry=registry,
            container_name=container_name,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
        try:
            try:
                if timeout_seconds is None:
                    stdout, stderr = process.communicate()
                else:
                    stdout, stderr = process.communicate(
                        timeout=timeout_seconds
                    )
            except BaseException as exc:
                errors = registry.terminate(process, container_name)
                try:
                    registry.ensure_accepting()
                except RehearsalTimeBudgetExceeded as cancelled:
                    _annotate_worker_cleanup_errors(cancelled, errors)
                    raise cancelled from exc
                _annotate_worker_cleanup_errors(exc, errors)
                raise
        finally:
            registry.unregister(process)
        registry.ensure_accepting()
        result = subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout,
            stderr,
        )
        if result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode,
                command,
                output=stdout,
                stderr=stderr,
            )
        return result
    return subprocess.run(
        list(args),
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=capture,
        timeout=timeout_seconds,
    )


def _git_sha(value: str) -> str:
    result = _run(
        ["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
        capture=True,
    )
    return result.stdout.strip()


def _git_file(commit_sha: str, path: str) -> bytes:
    args = ["git", "show", f"{commit_sha}:{path}"]
    registry = getattr(_WORKER_PROCESS_STATE, "registry", None)
    if registry is None:
        result = subprocess.run(
            args,
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            timeout=_GIT_BLOB_READ_TIMEOUT_SECONDS,
        )
        return result.stdout

    command, container_name = registry.prepare_command(args)
    process = _spawn_registered_process(
        command,
        registry=registry,
        container_name=container_name,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        try:
            stdout, stderr = process.communicate(
                timeout=_GIT_BLOB_READ_TIMEOUT_SECONDS
            )
        except BaseException as exc:
            errors = registry.terminate(process, container_name)
            try:
                registry.ensure_accepting()
            except RehearsalTimeBudgetExceeded as cancelled:
                _annotate_worker_cleanup_errors(cancelled, errors)
                raise cancelled from exc
            _annotate_worker_cleanup_errors(exc, errors)
            raise
    finally:
        registry.unregister(process)
    registry.ensure_accepting()
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return stdout


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in (0, 1):
        raise SystemExit("unable to resolve restart rehearsal Git ancestry")
    return result.returncode == 0


def _resolve_transition(
    from_sha: str,
    candidate_sha: str,
    requested: str,
) -> str:
    forward = _is_ancestor(from_sha, candidate_sha)
    rollback = (
        from_sha != candidate_sha
        and _is_ancestor(candidate_sha, from_sha)
    )
    if requested == "auto":
        if forward:
            return "forward"
        if rollback:
            return "rollback"
        raise SystemExit("restart rehearsal commits are unrelated")
    if requested == "forward" and not forward:
        raise SystemExit("forward rehearsal target does not descend from --from-sha")
    if requested == "rollback" and not rollback:
        raise SystemExit("rollback rehearsal target is not an ancestor of --from-sha")
    return requested


def _docker_platform(profile: str) -> str:
    if profile == "release":
        return "linux/amd64"
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "linux/arm64"
    if machine in {"amd64", "x86_64"}:
        return "linux/amd64"
    raise SystemExit(f"unsupported local Docker architecture: {machine}")


def _rehearsal_base_image(docker_platform: str) -> str:
    try:
        return REHEARSAL_BASE_IMAGES[docker_platform]
    except KeyError as exc:
        raise SystemExit(
            f"unsupported rehearsal Docker platform: {docker_platform}"
        ) from exc


def _inspect_local_image(reference: str) -> dict[str, Any] | None:
    try:
        result = _run(
            ["docker", "image", "inspect", reference],
            capture=True,
            timeout_seconds=_LOCAL_IMAGE_OPERATION_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        missing_messages = {
            f"Error response from daemon: No such image: {reference}",
            f"Error response from daemon: No such object: {reference}",
            f"Error: No such object: {reference}",
        }
        if str(exc.stderr or "").strip() in missing_messages:
            return None
        raise SystemExit("unable to inspect local Docker image") from exc
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SystemExit("local Docker image inspection is malformed") from exc
    if not isinstance(payload, list) or len(payload) != 1:
        raise SystemExit("local Docker image inspection is ambiguous")
    image = payload[0]
    if not isinstance(image, dict):
        raise SystemExit("local Docker image inspection is malformed")
    return image


def _prepare_local_rehearsal_base_image(docker_platform: str) -> str:
    """Bind the pinned base digest to a validated local no-pull alias.

    Resolve or pull the pinned platform image once, prove its immutable
    repository digest and platform, then build from a full-digest local alias
    with Docker's explicit no-pull policy.  An existing mismatched alias is
    treated as tampering instead of being overwritten.
    """

    pinned = _rehearsal_base_image(docker_platform)
    match = re.fullmatch(r".+@sha256:([0-9a-f]{64})", pinned)
    if match is None:
        raise SystemExit("rehearsal base image is not digest pinned")
    expected_digest = f"sha256:{match.group(1)}"
    expected_os, expected_arch = docker_platform.split("/", 1)

    image = _inspect_local_image(pinned)
    if image is None:
        _run(
            ["docker", "pull", "--platform", docker_platform, pinned],
            timeout_seconds=_REHEARSAL_BASE_PULL_TIMEOUT_SECONDS,
        )
        image = _inspect_local_image(pinned)
    if image is None:
        raise SystemExit("pinned rehearsal base image is unavailable locally")

    repo_digests = image.get("RepoDigests")
    if not isinstance(repo_digests, list) or pinned not in repo_digests:
        raise SystemExit("local rehearsal base repository digest differs")
    if image.get("Os") != expected_os or image.get("Architecture") != expected_arch:
        raise SystemExit("local rehearsal base platform differs")
    image_id = image.get("Id")
    if not isinstance(image_id, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_id
    ) is None:
        raise SystemExit("local rehearsal base image ID is malformed")

    alias = (
        f"{IMAGE_REPOSITORY}-base:"
        f"{expected_arch}-{expected_digest.removeprefix('sha256:')}"
    )
    alias_image = _inspect_local_image(alias)
    if alias_image is not None and alias_image.get("Id") != image_id:
        raise SystemExit("local rehearsal base alias differs from pinned image")
    if alias_image is None:
        _run(
            ["docker", "image", "tag", pinned, alias],
            timeout_seconds=_LOCAL_IMAGE_OPERATION_TIMEOUT_SECONDS,
        )
        alias_image = _inspect_local_image(alias)
    if alias_image is None or alias_image.get("Id") != image_id:
        raise SystemExit("local rehearsal base alias was not installed exactly")
    alias_repo_digests = alias_image.get("RepoDigests")
    if not isinstance(alias_repo_digests, list) or pinned not in alias_repo_digests:
        raise SystemExit("local rehearsal base alias lost the pinned digest")
    if (
        alias_image.get("Os") != expected_os
        or alias_image.get("Architecture") != expected_arch
    ):
        raise SystemExit("local rehearsal base alias platform differs")
    # The outer controller lock serializes this validation and the immediately
    # following build across every canonical rehearsal.  A plain local tag is
    # deliberately used here because classic Docker stores cannot resolve a
    # newly tagged image as alias@digest without first pushing that repository;
    # the exact ID and original pinned RepoDigest above are the local authority.
    return alias


def _image_tag(
    harness_sha: str,
    *,
    docker_platform: str,
    wheelhouse_shas: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"harness_sha")
    digest.update(harness_sha.encode("ascii"))
    digest.update(b"docker_platform")
    digest.update(docker_platform.encode("ascii"))
    digest.update(b"requirements.txt")
    digest.update(_git_file(harness_sha, "requirements.txt"))
    for wheelhouse_sha in sorted(set(wheelhouse_shas)):
        digest.update(b"wheelhouse_sha")
        digest.update(wheelhouse_sha.encode("ascii"))
        for path in SCORING_WHEELHOUSE_PATHS:
            digest.update(path.encode("utf-8"))
            digest.update(_git_file(wheelhouse_sha, path))
    for path in EXTERNAL_ARTIFACT_LOCK_PATHS:
        digest.update(path.encode("utf-8"))
        digest.update(_git_file(harness_sha, path))
    for path in COMMITTED_HARNESS_PATHS:
        digest.update(path.encode("utf-8"))
        digest.update(_git_file(harness_sha, path))
    return f"{IMAGE_REPOSITORY}:{digest.hexdigest()[:16]}"


def _build_image(
    tag: str,
    *,
    harness_sha: str,
    docker_platform: str,
    buildx_executable: Path,
    wheelhouse_shas: Sequence[str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="leadpoet-restart-image-") as raw:
        context = Path(raw)
        (context / "requirements.txt").write_bytes(
            _git_file(harness_sha, "requirements.txt")
        )
        (context / "Dockerfile").write_bytes(
            _git_file(
                harness_sha,
                "tests/restart_rehearsal/Dockerfile",
            )
        )
        scoring_locks = context / "scoring-locks"
        scoring_locks.mkdir()
        scoring_lock_payloads: dict[str, bytes] = {}
        scoring_lock_aliases: dict[str, str] = {}
        for wheelhouse_sha in sorted(set(wheelhouse_shas)):
            payload = _git_file(
                wheelhouse_sha,
                "gateway/tee/requirements-scoring-py39.lock",
            )
            lock_sha256 = hashlib.sha256(payload).hexdigest()
            scoring_lock_payloads[lock_sha256] = payload
            scoring_lock_aliases[wheelhouse_sha] = lock_sha256
        candidate_scoring_lock = _git_file(
            harness_sha,
            "gateway/tee/requirements-scoring-py39.lock",
        )
        candidate_scoring_lock_sha256 = hashlib.sha256(
            candidate_scoring_lock
        ).hexdigest()
        scoring_lock_payloads[candidate_scoring_lock_sha256] = (
            candidate_scoring_lock
        )
        scoring_lock_aliases[harness_sha] = candidate_scoring_lock_sha256
        for lock_sha256, payload in sorted(scoring_lock_payloads.items()):
            (scoring_locks / f"{lock_sha256}.lock").write_bytes(payload)
        (context / "scoring-lock-aliases.json").write_text(
            json.dumps(
                scoring_lock_aliases,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        for path in EXTERNAL_ARTIFACT_LOCK_PATHS:
            (context / Path(path).name).write_bytes(
                _git_file(harness_sha, path)
            )
        harness = context / "harness"
        harness.mkdir()
        for path in COMMITTED_HARNESS_PATHS[1:]:
            relative = Path(path).relative_to("tests/restart_rehearsal")
            destination = harness / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_git_file(harness_sha, path))
        base_image = _prepare_local_rehearsal_base_image(docker_platform)
        _run(
            [
                str(buildx_executable),
                "build",
                "--builder",
                "default",
                "--output",
                "type=docker,compression=zstd,compression-level=1,"
                "force-compression=true",
                "--progress=plain",
                "--pull=false",
                "--platform",
                docker_platform,
                "--build-arg",
                "REHEARSAL_BASE_IMAGE=" + base_image,
                "--build-arg",
                "REHEARSAL_SCORING_LOCK_SHA256="
                + candidate_scoring_lock_sha256,
                "--tag",
                tag,
                ".",
            ],
            cwd=context,
        )


def _verify_driver_identity(harness_sha: str) -> None:
    path = "scripts/run_local_restart_rehearsal.py"
    if Path(__file__).resolve().read_bytes() != _git_file(harness_sha, path):
        raise SystemExit(
            "restart rehearsal driver differs from the frozen harness SHA"
        )


def _populate_isolated_source_snapshot(
    *,
    source: Path,
    harness_sha: str,
    required_shas: Sequence[str],
) -> None:
    """Populate one frozen source root using registry-owned subprocesses."""

    _run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            "--no-checkout",
            str(REPO_ROOT),
            str(source),
        ]
    )
    for commit in required_shas:
        try:
            _run(
                [
                    "git",
                    "-C",
                    str(source),
                    "cat-file",
                    "-e",
                    f"{commit}^{{commit}}",
                ],
                capture=True,
            )
        except subprocess.CalledProcessError:
            _run(
                [
                    "git",
                    "-C",
                    str(source),
                    "fetch",
                    "--quiet",
                    str(REPO_ROOT),
                    commit,
                ]
            )
        _run(
            [
                "git",
                "-C",
                str(source),
                "cat-file",
                "-e",
                f"{commit}^{{commit}}",
            ]
        )
    _run(
        [
            "git",
            "-C",
            str(source),
            "checkout",
            "--quiet",
            "--detach",
            harness_sha,
        ]
    )
    _run(
        [
            "git",
            "-C",
            str(source),
            "fsck",
            "--strict",
            "--no-dangling",
        ]
    )


@contextmanager
def _isolated_source_snapshot_root() -> Iterator[Path]:
    """Own the source directory independently from its population schedule."""

    with tempfile.TemporaryDirectory(
        prefix="leadpoet-restart-source-"
    ) as raw:
        yield Path(raw) / "source"


@contextmanager
def _isolated_source_snapshot(
    *,
    harness_sha: str,
    required_shas: Sequence[str],
) -> Iterator[Path]:
    """Copy frozen Git objects so sequential containers cannot share mutations."""

    with _isolated_source_snapshot_root() as source:
        _populate_isolated_source_snapshot(
            source=source,
            harness_sha=harness_sha,
            required_shas=required_shas,
        )
        yield source


def _image_exists(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _drand_source_contract(lock_bytes: bytes) -> dict[str, str]:
    try:
        document = json.loads(lock_bytes)
    except (TypeError, ValueError) as exc:
        raise SystemExit("candidate drand source lock is malformed") from exc
    artifacts = document.get("artifacts") if isinstance(document, dict) else None
    contract = (
        artifacts.get("bittensor_drand_source")
        if isinstance(artifacts, dict)
        else None
    )
    if (
        not isinstance(document, dict)
        or document.get("schema_version")
        != "leadpoet.validator_runtime_artifacts.v2"
        or not isinstance(contract, dict)
        or set(contract) != {"filename", "sha256", "url"}
    ):
        raise SystemExit("candidate drand source lock is incomplete")
    filename = contract.get("filename")
    expected_sha256 = contract.get("sha256")
    url = contract.get("url")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename in {"", ".", ".."}
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
        )
        or not isinstance(url, str)
        or not url.startswith("https://")
    ):
        raise SystemExit("candidate drand source lock is invalid")
    return {
        "filename": filename,
        "sha256": expected_sha256,
        "url": url,
    }


def _download_locked_drand_source(argv: Sequence[str]) -> int:
    """Run the public download in a separately cancellable exact process."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    lock_path = args.lock.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    if not lock_path.is_file() or not output_root.is_dir():
        raise SystemExit("drand download paths are invalid")
    contract = _drand_source_contract(lock_path.read_bytes())
    destination = output_root / contract["filename"]
    partial = output_root / (
        f".{contract['filename']}.partial-{os.getpid()}"
    )
    request = Request(
        contract["url"],
        headers={"User-Agent": "leadpoet-local-restart-rehearsal/1"},
    )
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(
            partial,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with (
            os.fdopen(descriptor, "wb") as output,
            urlopen(
                request,
                timeout=_DRAND_SOURCE_DOWNLOAD_TIMEOUT_SECONDS,
            ) as response,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > _DRAND_SOURCE_MAX_BYTES:
                    raise SystemExit("candidate drand source archive is oversized")
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != contract["sha256"]:
            raise SystemExit("candidate drand source archive hash differs")
        os.replace(partial, destination)
        destination.chmod(0o644)
    finally:
        partial.unlink(missing_ok=True)
    return 0


def _drand_builder_tag(source_root: Path) -> str:
    dockerfile = source_root / "validator_tee/Dockerfile.drand-builder"
    digest = hashlib.sha256()
    digest.update(b"leadpoet.rehearsal.drand_builder.v1")
    digest.update(dockerfile.read_bytes())
    digest.update(b"linux/amd64")
    return f"validator-drand-builder:rehearsal-{digest.hexdigest()[:16]}"


def _publish_drand_cache(
    *, source: Path, destination: Path, expected_hash: str
) -> None:
    registry = getattr(_WORKER_PROCESS_STATE, "registry", None)
    if registry is None:
        raise RuntimeError("drand cache publication lacks process ownership")
    temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}-{threading.get_ident()}.tmp"
    )
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
        )
        with source.open("rb") as input_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            descriptor = -1
            shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_hash:
            raise SystemExit("published drand C ABI cache hash differs")
        temporary.chmod(0o444)
        registry.ensure_accepting()
        os.replace(temporary, destination)
        registry.ensure_accepting()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _prepare_drand_artifact_owned(
    *,
    source_root: Path,
    candidate_sha: str,
    buildx_executable: Path,
) -> Path:
    """Independently rebuild/cache the real measured drand C ABI artifact."""

    expected_hash = _git_file(
        candidate_sha,
        "validator_tee/enclave/libbittensor_drand_v2.sha256",
    ).decode("ascii").strip()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise SystemExit("candidate drand C ABI hash is invalid")
    cache_root = (
        Path.home()
        / ".cache"
        / "leadpoet-local-restart-rehearsal"
        / "drand-cabi-v2"
        / expected_hash
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / "libbittensor_drand_v2.so"

    def valid(path: Path) -> bool:
        return (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        )

    if valid(cached):
        return cache_root
    local = REPO_ROOT / ".validator-tee-artifacts/libbittensor_drand_v2.so"
    if valid(local):
        _publish_drand_cache(
            source=local,
            destination=cached,
            expected_hash=expected_hash,
        )
        return cache_root

    lock_bytes = _git_file(
        candidate_sha,
        "validator_tee/runtime-artifacts-v2.lock.json",
    )
    source_contract = _drand_source_contract(lock_bytes)
    artifact_root = source_root / ".validator-tee-artifacts"
    artifact_root.mkdir(exist_ok=True)
    source_archive = artifact_root / source_contract["filename"]
    with tempfile.TemporaryDirectory(
        prefix=".drand-cabi-v2.",
        dir=source_root,
    ) as raw:
        work_root = Path(raw)
        lock_path = work_root / "runtime-artifacts-v2.lock.json"
        lock_path.write_bytes(lock_bytes)
        lock_path.chmod(0o600)
        if (
            not source_archive.is_file()
            or hashlib.sha256(source_archive.read_bytes()).hexdigest()
            != source_contract["sha256"]
        ):
            _run(
                [
                    sys.executable,
                    str(source_root / "scripts/run_local_restart_rehearsal.py"),
                    _DRAND_INTERNAL_DOWNLOAD_COMMAND,
                    "--lock",
                    str(lock_path),
                    "--output-root",
                    str(artifact_root),
                ],
                cwd=source_root,
                timeout_seconds=_DRAND_SOURCE_DOWNLOAD_PROCESS_TIMEOUT_SECONDS,
            )
        if (
            not source_archive.is_file()
            or hashlib.sha256(source_archive.read_bytes()).hexdigest()
            != source_contract["sha256"]
        ):
            raise SystemExit("pinned drand source archive hash differs")

        shutil.copy2(source_archive, work_root / "source.tar.gz")
        shutil.copy2(
            source_root / "validator_tee/enclave/Cargo.drand-cabi-v2.lock",
            work_root / "Cargo.drand-cabi-v2.lock",
        )
        (work_root / "output").mkdir()
        cargo_cache = (
            Path.home() / ".cache" / "leadpoet" / "drand-cabi-v2"
        )
        (cargo_cache / "home").mkdir(parents=True, exist_ok=True)
        (cargo_cache / "target-al2-glibc226").mkdir(
            parents=True,
            exist_ok=True,
        )
        builder_tag = _drand_builder_tag(source_root)
        _run(
            [
                str(buildx_executable),
                "build",
                "--builder",
                "default",
                "--load",
                "--progress=plain",
                "--pull=false",
                "--platform",
                "linux/amd64",
                "-f",
                str(source_root / "validator_tee/Dockerfile.drand-builder"),
                "-t",
                builder_tag,
                str(source_root),
            ],
            cwd=source_root,
            timeout_seconds=_DRAND_BUILDER_TIMEOUT_SECONDS,
        )
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--network",
                "bridge",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{work_root}:/work",
                "-v",
                f"{cargo_cache}:/cargo-cache",
                "-v",
                f"{source_root}:/source:ro",
                builder_tag,
                "bash",
                "/source/validator_tee/scripts/build_drand_cabi_v2.sh",
                "--internal-no-docker",
            ],
            cwd=source_root,
            timeout_seconds=_DRAND_COMPILE_TIMEOUT_SECONDS,
        )
        output = work_root / "output/libbittensor_drand_v2.so"
        if not valid(output):
            raise SystemExit(
                "real drand C ABI rebuild differs from candidate hash"
            )
        _publish_drand_cache(
            source=output,
            destination=cached,
            expected_hash=expected_hash,
        )
    return cache_root


def _prepare_drand_artifact(
    *,
    source_root: Path,
    candidate_sha: str,
    buildx_executable: Path,
) -> Path:
    registry = getattr(_WORKER_PROCESS_STATE, "registry", None)
    if registry is not None:
        return _prepare_drand_artifact_owned(
            source_root=source_root,
            candidate_sha=candidate_sha,
            buildx_executable=buildx_executable,
        )
    owned_registry = _WorkerProcessRegistry()
    with _worker_process_scope(owned_registry):
        try:
            return _prepare_drand_artifact_owned(
                source_root=source_root,
                candidate_sha=candidate_sha,
                buildx_executable=buildx_executable,
            )
        except BaseException as exc:
            _annotate_worker_cleanup_errors(exc, owned_registry.cancel())
            raise


def _normalize_evidence_ownership(
    tag: str,
    *,
    evidence_root: Path,
    docker_platform: str,
) -> list[dict[str, Any]]:
    """Make descendants host-accessible without changing their ownership."""

    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        docker_platform,
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "128m",
        "--pids-limit",
        "32",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        # The host owns the mode-0700 bind root while candidate containers can
        # own its descendants.  Request only the traversal/chmod capabilities;
        # a daemon may still reject them under its user-namespace mapping, so
        # the host verifier below remains the sole handoff authority.
        "--cap-add",
        "DAC_OVERRIDE",
        "--cap-add",
        "FOWNER",
        "--read-only",
        "--mount",
        f"type=bind,src={evidence_root},dst=/evidence",
        "--entrypoint",
        "/usr/bin/find",
        tag,
        "/evidence",
        "-xdev",
        "-mindepth",
        "1",
        "(",
        "-type",
        "d",
        "-o",
        "(",
        "-type",
        "f",
        "-links",
        "1",
        ")",
        ")",
        "-exec",
        "/usr/bin/chmod",
        "--",
        "a+rwX",
        "{}",
        "+",
    ]
    deadline = time.monotonic() + _EVIDENCE_NORMALIZATION_TIMEOUT_SECONDS
    last_container_failure: BaseException | None = None
    last_host_failure: BaseException | None = None
    for _pass in range(_EVIDENCE_NORMALIZATION_MAX_PASSES):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            last_container_failure = subprocess.TimeoutExpired(
                command,
                _EVIDENCE_NORMALIZATION_TIMEOUT_SECONDS,
            )
            break
        container_failure: BaseException | None = None
        try:
            _run(
                command,
                capture=True,
                timeout_seconds=remaining,
            )
        except (KeyboardInterrupt, RehearsalTimeBudgetExceeded) as exc:
            _report_evidence_normalization_failure(
                phase="container",
                error=exc,
            )
            raise
        except BaseException as exc:
            container_failure = exc
            last_container_failure = exc

        try:
            _verify_host_evidence_access(evidence_root)
        except (KeyboardInterrupt, RehearsalTimeBudgetExceeded) as exc:
            _report_evidence_normalization_failure(phase="host", error=exc)
            raise
        except BaseException as exc:
            last_host_failure = exc
            if (
                _normalization_failure_category(exc) == "permission"
                and _pass + 1 < _EVIDENCE_NORMALIZATION_MAX_PASSES
                and time.monotonic() < deadline
            ):
                continue
            break
        else:
            if container_failure is None:
                return []
            reported = _report_evidence_normalization_failure(
                phase="container",
                error=container_failure,
            )
            return reported.diagnostics

    diagnostics: list[dict[str, Any]] = []
    if last_container_failure is not None:
        reported = _report_evidence_normalization_failure(
            phase="container",
            error=last_container_failure,
        )
        diagnostics.extend(reported.diagnostics)
    if last_host_failure is None:
        last_host_failure = TimeoutError("host evidence verification timed out")
    reported = _report_evidence_normalization_failure(
        phase="host",
        error=last_host_failure,
    )
    reported.diagnostics = [*diagnostics, *reported.diagnostics]
    raise reported from None


def _normalization_failure_category(error: BaseException) -> str:
    if isinstance(error, PermissionError) or (
        isinstance(error, OSError)
        and error.errno in {errno.EACCES, errno.EPERM, errno.EROFS}
    ):
        return "permission"
    if isinstance(error, FileNotFoundError) or (
        isinstance(error, OSError) and error.errno == errno.ENOENT
    ):
        return "not_found"
    if isinstance(
        error,
        (MemoryError, subprocess.TimeoutExpired, RehearsalTimeBudgetExceeded),
    ) or (
        isinstance(error, OSError)
        and error.errno
        in {errno.EAGAIN, errno.EMFILE, errno.ENFILE, errno.ENOMEM, errno.ENOSPC}
    ):
        return "resource"
    diagnostic = "\n".join(
        value.lower()
        for value in (
            str(getattr(error, "stderr", "") or ""),
            str(getattr(error, "stdout", "") or ""),
        )
    )
    if "permission denied" in diagnostic or "operation not permitted" in diagnostic:
        return "permission"
    if "no such file" in diagnostic or "not found" in diagnostic:
        return "not_found"
    if any(
        marker in diagnostic
        for marker in (
            "cannot allocate memory",
            "no space left",
            "out of memory",
            "resource temporarily unavailable",
        )
    ):
        return "resource"
    return "unknown"


def _normalization_failure_status(error: BaseException) -> int:
    if isinstance(error, subprocess.CalledProcessError):
        returncode = error.returncode
        if isinstance(returncode, int):
            if 1 <= returncode <= 255:
                return returncode
            if -127 <= returncode < 0:
                return 128 + abs(returncode)
    if isinstance(error, KeyboardInterrupt):
        return 130
    if isinstance(
        error,
        (subprocess.TimeoutExpired, RehearsalTimeBudgetExceeded),
    ):
        return 124
    if isinstance(error, FileNotFoundError):
        return 127
    if isinstance(error, PermissionError):
        return 126
    return 1


def _report_evidence_normalization_failure(
    *,
    phase: str,
    error: BaseException,
) -> _EvidenceNormalizationError:
    category = _normalization_failure_category(error)
    status = _normalization_failure_status(error)
    try:
        print(
            "REHEARSAL_EVIDENCE_NORMALIZATION_FAILED "
            f"phase={phase} category={category} status={status}",
            file=sys.stderr,
            flush=True,
        )
    except BaseException:
        pass
    return _EvidenceNormalizationError(
        phase=phase,
        category=category,
        status=status,
    )


def _report_fixture_generation_failure(error: BaseException) -> dict[str, Any]:
    category = _normalization_failure_category(error)
    status = _normalization_failure_status(error)
    try:
        print(
            "REHEARSAL_FIXTURE_GENERATION_FAILED "
            f"category={category} status={status}",
            file=sys.stderr,
            flush=True,
        )
    except BaseException:
        pass
    return {"category": category, "status": status}


def _attach_normalization_diagnostics(
    error: BaseException,
    diagnostics: Sequence[dict[str, Any]],
) -> None:
    if not diagnostics:
        return
    try:
        existing = getattr(
            error,
            "_rehearsal_evidence_normalization_diagnostics",
            [],
        )
        if not isinstance(existing, list):
            existing = []
        setattr(
            error,
            "_rehearsal_evidence_normalization_diagnostics",
            [*existing, *diagnostics],
        )
    except BaseException:
        pass


def _verify_host_evidence_access(evidence_root: Path) -> None:
    """Prove the private root and every supported descendant are accessible."""

    root_stat = evidence_root.lstat()
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != os.geteuid()
        or stat.S_IMODE(root_stat.st_mode) != 0o700
    ):
        raise PermissionError("evidence root identity or mode differs")

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current, directory_names, file_names in os.walk(
        evidence_root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in (*directory_names, *file_names):
            path = current_path / name
            path_stat = path.lstat()
            if stat.S_ISLNK(path_stat.st_mode):
                raise OSError("evidence tree contains a symbolic link")
            if path_stat.st_dev != root_stat.st_dev:
                raise OSError("evidence tree crosses a device boundary")
            mode = stat.S_IMODE(path_stat.st_mode)
            if stat.S_ISDIR(path_stat.st_mode):
                if mode & 0o007 != 0o007 or not os.access(
                    path,
                    os.R_OK | os.W_OK | os.X_OK,
                ):
                    raise PermissionError("evidence directory is inaccessible")
                continue
            if stat.S_ISREG(path_stat.st_mode):
                if path_stat.st_nlink != 1:
                    raise OSError("evidence tree contains a hard-linked file")
                if mode & 0o006 != 0o006 or not os.access(
                    path,
                    os.R_OK | os.W_OK,
                ):
                    raise PermissionError("evidence file is inaccessible")
                continue
            raise OSError("evidence tree contains an unsupported file type")


def _workflow_projection_stage_kind(stage: str) -> str:
    if stage == "input-contract":
        return "input"
    if stage == "production-allocation-input":
        return "allocation"
    if stage.startswith("source-identity:"):
        return "source_identity"
    if stage.startswith("behavior:"):
        return "behavior"
    if stage.startswith("diagnostic:"):
        return "diagnostic"
    if stage.startswith("fault:"):
        return "fault"
    if stage == "concurrency":
        return "concurrency"
    if stage in {"boundary-start", "boundary-cleanup"}:
        return "boundary"
    if re.fullmatch(r"epoch-[0-9]+", stage):
        return "epoch"
    if stage == "workflow-evidence-validation":
        return "validation"
    return "unknown"


def _print_safe_rehearsal_marker(marker: str) -> None:
    try:
        print(marker, file=sys.stderr, flush=True)
    except BaseException:
        pass


def _project_workflow_failure_diagnostics(
    *,
    evidence_root: Path,
    candidate_sha: str,
    profile: str,
) -> dict[str, Any]:
    """Emit and return a bounded, hash-only workflow failure projection."""

    try:
        path = evidence_root / "workflow.json"
        root_stat = evidence_root.lstat()
        path_stat = path.lstat()
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or path_stat.st_dev != root_stat.st_dev
            or not 0 < path_stat.st_size <= _WORKFLOW_FAILURE_PROJECTION_MAX_BYTES
        ):
            raise ValueError("workflow evidence file identity differs")
        document = json.loads(path.read_bytes().decode("utf-8"))
        if (
            not isinstance(document, dict)
            or document.get("schema_version")
            != "leadpoet.local_v2_workflow_evidence.v1"
            or document.get("status") != "failed"
            or document.get("release_sha") != candidate_sha
            or document.get("profile") != profile
        ):
            raise ValueError("workflow evidence identity differs")
        raw_stages = document.get("stages")
        if not isinstance(raw_stages, list) or len(raw_stages) > 512:
            raise ValueError("workflow evidence stage set is invalid")

        projected: list[dict[str, Any]] = []
        for item in raw_stages:
            if not isinstance(item, dict):
                raise ValueError("workflow evidence stage is invalid")
            status = item.get("status")
            if status not in {"failed", "unexercised"}:
                continue
            stage = item.get("stage")
            if not isinstance(stage, str) or not 0 < len(stage) <= 1024:
                raise ValueError("workflow evidence stage identity is invalid")
            error_type = "None"
            if status == "failed":
                raw_error_type = item.get("error_type")
                error_type = (
                    raw_error_type
                    if raw_error_type in _SAFE_WORKFLOW_PROJECTION_ERROR_TYPES
                    else "OtherError"
                )
            projected.append(
                {
                    "error_type": error_type,
                    "stage_id_sha256": hashlib.sha256(
                        stage.encode("utf-8")
                    ).hexdigest(),
                    "stage_kind": _workflow_projection_stage_kind(stage),
                    "status": status,
                }
            )

        ordered = [
            item for item in projected if item["status"] == "failed"
        ] + [item for item in projected if item["status"] == "unexercised"]
        emitted = ordered[:_WORKFLOW_FAILURE_PROJECTION_MAX_STAGES]
        projection = {
            "available": True,
            "emitted_count": len(emitted),
            "failed_count": sum(
                item["status"] == "failed" for item in projected
            ),
            "stages": emitted,
            "truncated": len(emitted) != len(projected),
            "unexercised_count": sum(
                item["status"] == "unexercised" for item in projected
            ),
        }
    except (KeyboardInterrupt, RehearsalTimeBudgetExceeded):
        raise
    except Exception as exc:
        category = _normalization_failure_category(exc)
        status = _normalization_failure_status(exc)
        projection = {
            "available": False,
            "category": category,
            "status": status,
        }
        _print_safe_rehearsal_marker(
            "REHEARSAL_WORKFLOW_DIAGNOSTIC_UNAVAILABLE "
            f"category={category} status={status}"
        )
        return projection

    _print_safe_rehearsal_marker(
        "REHEARSAL_WORKFLOW_FAILURE_SUMMARY "
        f"failed={projection['failed_count']} "
        f"unexercised={projection['unexercised_count']} "
        f"emitted={projection['emitted_count']} "
        f"truncated={int(projection['truncated'])}"
    )
    for item in projection["stages"]:
        _print_safe_rehearsal_marker(
            "REHEARSAL_WORKFLOW_STAGE_RESULT "
            f"status={item['status']} "
            f"stage_kind={item['stage_kind']} "
            f"stage_id_sha256={item['stage_id_sha256']} "
            f"error_type={item['error_type']}"
        )
    return projection


def _normalize_evidence_after_failure(
    tag: str,
    *,
    evidence_root: Path,
    docker_platform: str,
    original: BaseException,
) -> bool:
    """Best-effort ownership handoff without replacing the source failure."""

    try:
        diagnostics = _normalize_evidence_ownership(
            tag,
            evidence_root=evidence_root,
            docker_platform=docker_platform,
        )
    except (KeyboardInterrupt, RehearsalTimeBudgetExceeded):
        raise
    except BaseException as cleanup_exc:
        if isinstance(cleanup_exc, _EvidenceNormalizationError):
            _attach_normalization_diagnostics(
                original,
                cleanup_exc.diagnostics,
            )
        _annotate_worker_cleanup_errors(
            original,
            [f"normalize:{type(cleanup_exc).__name__}"],
        )
        return False
    _attach_normalization_diagnostics(original, diagnostics)
    return True


@contextmanager
def _temporary_evidence_directory(
    tag: str,
    *,
    docker_platform: str,
    handoff_attempted: Callable[[], bool] | None = None,
) -> Iterator[Path]:
    """Normalize after the last possible writer and before host cleanup."""

    evidence_root = Path(
        tempfile.mkdtemp(prefix="leadpoet-restart-evidence-")
    )

    def report_retained(reason: BaseException) -> None:
        try:
            print(
                "REHEARSAL_EVIDENCE_RETAINED "
                f"path={evidence_root} "
                f"error_type={type(reason).__name__} "
                "error='normalization-or-delete-failed'",
                file=sys.stderr,
                flush=True,
            )
        except BaseException:
            pass

    def normalize() -> None:
        if handoff_attempted is not None and handoff_attempted():
            return
        registry = _WorkerProcessRegistry()
        with _worker_process_scope(registry):
            _normalize_evidence_ownership(
                tag,
                evidence_root=evidence_root,
                docker_platform=docker_platform,
            )

    try:
        yield evidence_root
    except BaseException as original:
        cleanup_errors = []
        try:
            normalize()
        except (KeyboardInterrupt, RehearsalTimeBudgetExceeded):
            raise
        except BaseException as cleanup_exc:
            cleanup_errors.append(
                f"normalize:{type(cleanup_exc).__name__}"
            )
            report_retained(cleanup_exc)
        else:
            try:
                shutil.rmtree(evidence_root)
            except (KeyboardInterrupt, RehearsalTimeBudgetExceeded):
                raise
            except BaseException as cleanup_exc:
                cleanup_errors.append(
                    f"delete:{type(cleanup_exc).__name__}"
                )
                report_retained(cleanup_exc)
        _annotate_worker_cleanup_errors(original, cleanup_errors)
        raise
    else:
        try:
            normalize()
            shutil.rmtree(evidence_root)
        except BaseException as cleanup_exc:
            report_retained(cleanup_exc)
            raise


def _run_component(
    tag: str,
    *,
    source_root: Path,
    component: str,
    from_sha: str,
    candidate_sha: str,
    transition: str,
    evidence_root: Path,
    drand_artifact_root: Path,
    profile: str,
    weight_readiness_scenario: str = "production_success",
    docker_platform: str,
    fixture_seed_root: Path,
    from_fixture_seed_root: Path,
    durable_fixture_seed_root: Path,
    durable_state_root: Path,
    durable_schema_sha: str,
    run_ordinal: int,
    gateway_worker_fleet_mode: str,
) -> None:
    limits = PROFILE_LIMITS[profile]
    launcher_log = evidence_root / (
        f"{run_ordinal}-{component}-{transition}-{candidate_sha}-launcher.log"
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        docker_platform,
        "--network",
        "none",
        "--cpus",
        str(limits["cpus"]),
        "--memory",
        str(limits["memory"]),
        "--pids-limit",
        "2048",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=2g",
        "--mount",
        f"type=bind,src={source_root},dst=/source,readonly",
        *_SOURCE_GIT_SAFE_DIRECTORY_DOCKER_ARGS,
        "--mount",
        f"type=bind,src={evidence_root},dst=/evidence",
        "--mount",
        (
            f"type=bind,src={drand_artifact_root},"
            "dst=/opt/leadpoet/drand-cabi-v2,readonly"
        ),
        "--mount",
        (
            f"type=bind,src={fixture_seed_root},"
            "dst=/rehearsal-fixture-seed,readonly"
        ),
        "--mount",
        (
            f"type=bind,src={from_fixture_seed_root},"
            "dst=/rehearsal-from-fixture-seed,readonly"
        ),
        "--mount",
        (
            f"type=bind,src={durable_fixture_seed_root},"
            "dst=/rehearsal-durable-schema-seed,readonly"
        ),
        "--mount",
        (
            f"type=bind,src={durable_state_root},"
            "dst=/rehearsal-durable-state"
        ),
        "--env",
        f"REHEARSAL_COMPONENT={component}",
        "--env",
        f"REHEARSAL_FROM_SHA={from_sha}",
        "--env",
        f"REHEARSAL_CANDIDATE_SHA={candidate_sha}",
        "--env",
        f"REHEARSAL_TRANSITION={transition}",
        "--env",
        (
            "REHEARSAL_WEIGHT_READINESS_SCENARIO="
            f"{weight_readiness_scenario}"
        ),
        "--env",
        "REHEARSAL_SCOPE=exact",
        "--env",
        f"REHEARSAL_PROFILE={profile}",
        "--env",
        f"REHEARSAL_RUN_ORDINAL={run_ordinal}",
        "--env",
        f"REHEARSAL_DURABLE_SCHEMA_SHA={durable_schema_sha}",
        "--env",
        f"REHEARSAL_GATEWAY_WORKER_FLEET_MODE={gateway_worker_fleet_mode}",
        tag,
    ]
    registry = getattr(_WORKER_PROCESS_STATE, "registry", None)
    container_name = None
    if registry is not None:
        command, container_name = registry.prepare_command(command)
    try:
        with launcher_log.open("w", encoding="utf-8") as output:
            process = _spawn_registered_process(
                command,
                registry=registry,
                container_name=container_name,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
            except BaseException as exc:
                errors = []
                try:
                    if registry is None:
                        _terminate_worker_process(process)
                    else:
                        errors.extend(
                            registry.terminate(process, container_name)
                        )
                except BaseException as cleanup_exc:
                    errors.append(
                        "subprocess:"
                        f"{type(cleanup_exc).__name__}:{cleanup_exc}"
                    )
                if registry is not None:
                    try:
                        registry.ensure_accepting()
                    except RehearsalTimeBudgetExceeded as cancelled:
                        _annotate_worker_cleanup_errors(cancelled, errors)
                        raise cancelled from exc
                _annotate_worker_cleanup_errors(exc, errors)
                raise
            finally:
                if registry is not None:
                    registry.unregister(process)
            if registry is not None:
                registry.ensure_accepting()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    except (KeyboardInterrupt, RehearsalTimeBudgetExceeded):
        raise
    except BaseException:
        raise


def _preserve_batched_failure_evidence(
    *,
    evidence_root: Path,
    candidate_sha: str,
    stages: Sequence[dict[str, Any]],
) -> Path:
    """Preserve the complete full-path stage ledger after a failed run."""

    durable_root = Path(
        tempfile.mkdtemp(
            prefix=(
                "leadpoet-rehearsal-failure-"
                f"{candidate_sha[:12]}-full-path-"
            )
        )
    )
    copied_evidence = durable_root / "evidence"
    shutil.copytree(evidence_root, copied_evidence)
    failures = [item for item in stages if item.get("status") == "failed"]
    unexercised = [
        item for item in stages if item.get("status") == "unexercised"
    ]
    report = {
        "candidate_sha": candidate_sha,
        "failure_count": len(failures),
        "failures": failures,
        "stage_count": len(stages),
        "stages": list(stages),
        "status": "failed",
        "unexercised_count": len(unexercised),
    }
    (durable_root / "failure-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"REHEARSAL_BATCH_FAILURE_EVIDENCE {durable_root}",
        file=sys.stderr,
        flush=True,
    )
    return durable_root


def _stage_failure(
    *,
    stage: str,
    exc: subprocess.CalledProcessError,
) -> dict[str, Any]:
    return {
        "command": [str(item) for item in exc.cmd],
        "returncode": int(exc.returncode),
        "stage": stage,
    }


def _stage_result_from_exception(
    *,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "error": str(exc)[:2000],
        "error_type": type(exc).__name__,
        "stage": stage,
        "status": "failed",
    }
    if isinstance(exc, subprocess.CalledProcessError):
        result.update(_stage_failure(stage=stage, exc=exc))
        result["status"] = "failed"
    result.update(_safe_exception_diagnostic_projection(exc))
    return result


def _safe_exception_diagnostic_projection(
    exc: BaseException,
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    fixture = getattr(
        exc,
        "_rehearsal_fixture_generation_diagnostic",
        None,
    )
    if (
        isinstance(fixture, dict)
        and fixture.get("category")
        in {"permission", "not_found", "resource", "unknown"}
        and type(fixture.get("status")) is int
        and 1 <= fixture["status"] <= 255
    ):
        projection["fixture_generation_diagnostic"] = {
            "category": fixture["category"],
            "status": fixture["status"],
        }

    normalization = getattr(
        exc,
        "_rehearsal_evidence_normalization_diagnostics",
        None,
    )
    if isinstance(exc, _EvidenceNormalizationError):
        normalization = exc.diagnostics
    if isinstance(normalization, list) and len(normalization) <= 2:
        normalized: list[dict[str, Any]] = []
        for item in normalization:
            if (
                not isinstance(item, dict)
                or item.get("phase") not in {"container", "host"}
                or item.get("category")
                not in {"permission", "not_found", "resource", "unknown"}
                or type(item.get("status")) is not int
                or not 1 <= item["status"] <= 255
            ):
                normalized = []
                break
            normalized.append(
                {
                    "category": item["category"],
                    "phase": item["phase"],
                    "status": item["status"],
                }
            )
        if normalized:
            projection["evidence_normalization_diagnostics"] = normalized

    workflow = getattr(exc, "_rehearsal_workflow_projection", None)
    safe_workflow = _safe_workflow_projection(workflow)
    if safe_workflow is not None:
        projection["workflow_failure_projection"] = safe_workflow
    return projection


def _safe_workflow_projection(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or type(value.get("available")) is not bool:
        return None
    if value["available"] is False:
        if (
            value.get("category")
            not in {"permission", "not_found", "resource", "unknown"}
            or type(value.get("status")) is not int
            or not 1 <= value["status"] <= 255
        ):
            return None
        return {
            "available": False,
            "category": value["category"],
            "status": value["status"],
        }

    counts = (
        value.get("failed_count"),
        value.get("unexercised_count"),
        value.get("emitted_count"),
    )
    if (
        any(type(item) is not int or not 0 <= item <= 512 for item in counts)
        or type(value.get("truncated")) is not bool
        or not isinstance(value.get("stages"), list)
        or len(value["stages"]) > _WORKFLOW_FAILURE_PROJECTION_MAX_STAGES
        or value["emitted_count"] != len(value["stages"])
    ):
        return None
    stage_kinds = {
        "allocation",
        "behavior",
        "boundary",
        "concurrency",
        "diagnostic",
        "epoch",
        "fault",
        "input",
        "source_identity",
        "unknown",
        "validation",
    }
    stages: list[dict[str, Any]] = []
    for item in value["stages"]:
        if (
            not isinstance(item, dict)
            or item.get("status") not in {"failed", "unexercised"}
            or item.get("stage_kind") not in stage_kinds
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(item.get("stage_id_sha256") or ""),
            )
            is None
            or item.get("error_type")
            not in {*_SAFE_WORKFLOW_PROJECTION_ERROR_TYPES, "None", "OtherError"}
        ):
            return None
        stages.append(
            {
                "error_type": item["error_type"],
                "stage_id_sha256": item["stage_id_sha256"],
                "stage_kind": item["stage_kind"],
                "status": item["status"],
            }
        )
    return {
        "available": True,
        "emitted_count": value["emitted_count"],
        "failed_count": value["failed_count"],
        "stages": stages,
        "truncated": value["truncated"],
        "unexercised_count": value["unexercised_count"],
    }


def _emit_terminal_stage_diagnostics(
    stages: Sequence[dict[str, Any]],
) -> None:
    """Re-emit only bounded fixed diagnostics at the aggregate failure tail."""

    for item in stages:
        fixture = item.get("fixture_generation_diagnostic")
        if isinstance(fixture, dict):
            _print_safe_rehearsal_marker(
                "REHEARSAL_FIXTURE_GENERATION_FAILED "
                f"category={fixture['category']} status={fixture['status']}"
            )
        normalization = item.get("evidence_normalization_diagnostics")
        if isinstance(normalization, list):
            for diagnostic in normalization:
                _print_safe_rehearsal_marker(
                    "REHEARSAL_EVIDENCE_NORMALIZATION_FAILED "
                    f"phase={diagnostic['phase']} "
                    f"category={diagnostic['category']} "
                    f"status={diagnostic['status']}"
                )
        workflow = _safe_workflow_projection(
            item.get("workflow_failure_projection")
        )
        if workflow is None:
            continue
        if workflow["available"] is False:
            _print_safe_rehearsal_marker(
                "REHEARSAL_WORKFLOW_DIAGNOSTIC_UNAVAILABLE "
                f"category={workflow['category']} status={workflow['status']}"
            )
            continue
        _print_safe_rehearsal_marker(
            "REHEARSAL_WORKFLOW_FAILURE_SUMMARY "
            f"failed={workflow['failed_count']} "
            f"unexercised={workflow['unexercised_count']} "
            f"emitted={workflow['emitted_count']} "
            f"truncated={int(workflow['truncated'])}"
        )
        for diagnostic in workflow["stages"]:
            _print_safe_rehearsal_marker(
                "REHEARSAL_WORKFLOW_STAGE_RESULT "
                f"status={diagnostic['status']} "
                f"stage_kind={diagnostic['stage_kind']} "
                f"stage_id_sha256={diagnostic['stage_id_sha256']} "
                f"error_type={diagnostic['error_type']}"
            )


def _attach_terminal_workflow_failure_projection(
    *,
    stages: Sequence[dict[str, Any]],
    workflow_stage: str,
    evidence_root: Path,
    candidate_sha: str,
    profile: str,
) -> None:
    """Attach bounded workflow detail only after shared writers are terminal."""

    failed = [
        item
        for item in stages
        if item.get("stage") == workflow_stage and item.get("status") == "failed"
    ]
    if len(failed) > 1:
        raise RuntimeError("workflow stage result is duplicated")
    if not failed:
        return
    projection = _project_workflow_failure_diagnostics(
        evidence_root=evidence_root,
        candidate_sha=candidate_sha,
        profile=profile,
    )
    safe_projection = _safe_workflow_projection(projection)
    if safe_projection is None:
        raise RuntimeError("workflow failure projection is unsafe")
    failed[0]["workflow_failure_projection"] = safe_projection


def _complete_shared_evidence_handoff(
    tag: str,
    *,
    stages: Sequence[dict[str, Any]],
    workflow_stage: str,
    evidence_root: Path,
    candidate_sha: str,
    profile: str,
    docker_platform: str,
) -> None:
    """Verify the shared tree once, after every candidate writer has joined."""

    registry = _WorkerProcessRegistry()
    try:
        with _worker_process_scope(registry):
            _normalize_evidence_ownership(
                tag,
                evidence_root=evidence_root,
                docker_platform=docker_platform,
            )
    except BaseException as original:
        try:
            cleanup_errors = registry.cancel()
        except BaseException as cleanup_exc:
            cleanup_errors = (
                f"registry:{type(cleanup_exc).__name__}",
            )
        _annotate_worker_cleanup_errors(original, cleanup_errors)
        raise
    _attach_terminal_workflow_failure_projection(
        stages=stages,
        workflow_stage=workflow_stage,
        evidence_root=evidence_root,
        candidate_sha=candidate_sha,
        profile=profile,
    )


def _run_independent_stage(
    *,
    stage: str,
    action: Callable[[], Any],
    stages: list[dict[str, Any]],
) -> tuple[bool, Any]:
    """Run one stage without suppressing later independent diagnostics."""

    started = time.monotonic()
    try:
        value = action()
    except KeyboardInterrupt:
        raise
    except RehearsalTimeBudgetExceeded:
        raise
    except BaseException as exc:
        result = _stage_result_from_exception(stage=stage, exc=exc)
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        stages.append(result)
        print(
            "REHEARSAL_STAGE_FAILED_CONTINUING "
            f"stage={stage} error_type={result['error_type']} "
            f"duration_seconds={result['duration_seconds']} "
            f"error={result['error']!r}",
            file=sys.stderr,
            flush=True,
        )
        return False, None
    duration = round(time.monotonic() - started, 3)
    stages.append(
        {
            "duration_seconds": duration,
            "stage": stage,
            "status": "passed",
        }
    )
    print(
        f"REHEARSAL_STAGE_PASSED stage={stage} "
        f"duration_seconds={duration}",
        flush=True,
    )
    return True, value


_PREPUSH_PHASES = frozenset(
    (
        "exact-image-build",
        "fixture-preparation",
        "gateway-runtime",
        "runtime-prefix",
        "source-snapshot",
        "validator-runtime",
        "workflow-runtime",
    )
)


def _run_prepush_phase(
    *,
    phase: str,
    action: Callable[[], Any],
) -> Any:
    """Emit bounded diagnostics without exposing command or exception text."""

    if phase not in _PREPUSH_PHASES:
        raise ValueError(f"unsupported prepush phase: {phase!r}")
    started = time.monotonic()
    print(
        "REHEARSAL_PREPUSH_PHASE "
        f"phase={phase} status=started duration_seconds=0.0",
        file=sys.stderr,
        flush=True,
    )
    try:
        result = action()
    except BaseException:
        duration = round(time.monotonic() - started, 3)
        print(
            "REHEARSAL_PREPUSH_PHASE "
            f"phase={phase} status=failed duration_seconds={duration}",
            file=sys.stderr,
            flush=True,
        )
        raise
    duration = round(time.monotonic() - started, 3)
    print(
        "REHEARSAL_PREPUSH_PHASE "
        f"phase={phase} status=passed duration_seconds={duration}",
        file=sys.stderr,
        flush=True,
    )
    return result


def _run_prepush_image_and_probe(
    *,
    image_action: Callable[[], Any],
    probe_action: Callable[[], Any],
) -> tuple[Any, Any]:
    """Keep the exact image build alarm-bound while overlapping its probe."""

    registry = _WorkerProcessRegistry()

    def registered_probe() -> Any:
        with _worker_process_scope(registry):
            registry.ensure_accepting()
            result = probe_action()
            registry.ensure_accepting()
            return result

    with _signal_masked_worker_executor() as executor:
        try:
            probe_future = executor.submit(registered_probe)
            with _worker_process_scope(registry):
                registry.ensure_accepting()
                image_result = image_action()
                registry.ensure_accepting()
            probe_result = probe_future.result()
        except BaseException as exc:
            _annotate_worker_cleanup_errors(exc, registry.cancel())
            raise
    return image_result, probe_result


def _run_prepush_image_and_snapshot(
    *,
    image_action: Callable[[], Any],
    snapshot_action: Callable[[], Any],
) -> tuple[Any, Any]:
    """Build on main while one masked worker prepares the frozen source."""

    return _run_prepush_image_and_probe(
        image_action=lambda: _run_prepush_phase(
            phase="exact-image-build",
            action=image_action,
        ),
        probe_action=lambda: _run_prepush_phase(
            phase="source-snapshot",
            action=snapshot_action,
        ),
    )


def _run_prepush_image_snapshot_and_artifacts(
    *,
    image_action: Callable[[], Any],
    snapshot_action: Callable[[], Any],
    artifact_action: Callable[[], Any],
) -> tuple[Any, Any, Any]:
    """Populate the frozen source, then prepare artifacts under the image."""

    def prepare_source_and_artifacts() -> tuple[Any, Any]:
        snapshot_result = _run_prepush_phase(
            phase="source-snapshot",
            action=snapshot_action,
        )
        return snapshot_result, artifact_action()

    image_result, worker_result = _run_prepush_image_and_probe(
        image_action=lambda: _run_prepush_phase(
            phase="exact-image-build",
            action=image_action,
        ),
        probe_action=prepare_source_and_artifacts,
    )
    return image_result, worker_result[0], worker_result[1]


def _run_prepush_runtime_stages(
    *,
    preparation_action: Callable[
        [], Sequence[tuple[str, Callable[[], Any]]]
    ],
    workflow_action: tuple[str, Callable[[], Any]],
    expected_component_stages: Sequence[str],
    stages: list[dict[str, Any]],
    worker_prefix_action: tuple[str, Callable[[], Any]] | None = None,
    worker_prefix_stage_index: int | None = None,
) -> list[dict[str, Any]]:
    """Schedule probe -> fixture -> validator and workflow -> gateway.

    Workflow remains on the alarm-owning main thread. The sole worker prepares
    any independent prefix stage, fixtures, and then the longer validator
    action. Gateway uses the main slot when workflow settles. Ordinary failures
    do not suppress an independent stage, and component results retain their
    declared order. The worker's non-subprocess work is fixed-size fixture
    bookkeeping bracketed by cancellation checks; every blocking external
    operation is registry-owned.
    """

    if (
        len(expected_component_stages) != 2
        or len(set(expected_component_stages)) != 2
    ):
        raise ValueError("prepush scheduling requires two expected components")
    if worker_prefix_action is None:
        if worker_prefix_stage_index is not None:
            raise ValueError("prefix stage index requires a prefix action")
        preparation_stage_index = len(stages)
    else:
        preparation_stage_index = (
            len(stages)
            if worker_prefix_stage_index is None
            else worker_prefix_stage_index
        )
        if (
            type(preparation_stage_index) is not int
            or not 0 <= preparation_stage_index <= len(stages)
        ):
            raise ValueError("prefix stage index is outside the stage ledger")

    def run_stage(
        item: tuple[str, Callable[[], Any]],
    ) -> list[dict[str, Any]]:
        phase = (
            "validator-runtime"
            if item[0].startswith("validator-")
            else "gateway-runtime"
        )
        local_stages: list[dict[str, Any]] = []
        _run_independent_stage(
            stage=item[0],
            action=lambda: _run_prepush_phase(
                phase=phase,
                action=item[1],
            ),
            stages=local_stages,
        )
        return local_stages

    actions_ready = threading.Event()
    execution_actions: list[tuple[str, Callable[[], Any]]] = []
    component_results: dict[str, list[dict[str, Any]]] = {}
    preparation_results: list[dict[str, Any]] = []
    workflow_results: list[dict[str, Any]] = []

    def prepare_then_run_validator() -> None:
        started = time.monotonic()
        try:
            registry.ensure_accepting()
            if worker_prefix_action is not None:
                _run_independent_stage(
                    stage=worker_prefix_action[0],
                    action=lambda: _run_prepush_phase(
                        phase="runtime-prefix",
                        action=worker_prefix_action[1],
                    ),
                    stages=preparation_results,
                )
                registry.ensure_accepting()
            started = time.monotonic()
            prepared = list(
                _run_prepush_phase(
                    phase="fixture-preparation",
                    action=preparation_action,
                )
            )
            registry.ensure_accepting()
            stage_names = [item[0] for item in prepared]
            if (
                len(prepared) > 2
                or len(set(stage_names)) != len(stage_names)
                or (
                    prepared
                    and set(stage_names) != set(expected_component_stages)
                )
            ):
                raise ValueError("prepush preparation returned invalid components")
            execution_actions.extend(
                sorted(
                    prepared,
                    key=lambda item: (
                        not item[0].startswith("validator-"),
                        item[0],
                    ),
                )
            )
        except (KeyboardInterrupt, RehearsalTimeBudgetExceeded):
            raise
        except BaseException as exc:
            result = _stage_result_from_exception(
                stage="fixture-orchestration",
                exc=exc,
            )
            result["duration_seconds"] = round(time.monotonic() - started, 3)
            preparation_results.append(result)
            print(
                "REHEARSAL_STAGE_FAILED_CONTINUING "
                "stage=fixture-orchestration "
                f"error_type={result['error_type']} "
                f"duration_seconds={result['duration_seconds']} "
                f"error={result['error']!r}",
                file=sys.stderr,
                flush=True,
            )
            for stage in expected_component_stages:
                local_stages: list[dict[str, Any]] = []
                _mark_stage_unexercised(
                    stage=stage,
                    blocked_by=["fixture-orchestration"],
                    stages=local_stages,
                )
                component_results[stage] = local_stages
        finally:
            actions_ready.set()
        if execution_actions:
            registry.ensure_accepting()
            first = execution_actions[0]
            component_results[first[0]] = run_stage(first)

    registry = _WorkerProcessRegistry()

    def registered_preparation() -> None:
        with _worker_process_scope(registry):
            registry.ensure_accepting()
            prepare_then_run_validator()
            registry.ensure_accepting()

    with _signal_masked_worker_executor() as executor:
        try:
            preparation_future = executor.submit(registered_preparation)
            with _worker_process_scope(registry):
                registry.ensure_accepting()
                _run_independent_stage(
                    stage=workflow_action[0],
                    action=lambda: _run_prepush_phase(
                        phase="workflow-runtime",
                        action=workflow_action[1],
                    ),
                    stages=workflow_results,
                )
                actions_ready.wait()
                if len(execution_actions) == 2:
                    second = execution_actions[1]
                    component_results[second[0]] = run_stage(second)
                registry.ensure_accepting()
            preparation_future.result()
        except BaseException as exc:
            _annotate_worker_cleanup_errors(exc, registry.cancel())
            raise

    stages[preparation_stage_index:preparation_stage_index] = preparation_results
    for stage in expected_component_stages:
        if stage in component_results:
            stages.extend(component_results[stage])
    return workflow_results


def _mark_stage_unexercised(
    *,
    stage: str,
    blocked_by: Sequence[str],
    stages: list[dict[str, Any]],
) -> None:
    result = {
        "blocked_by": list(blocked_by),
        "stage": stage,
        "status": "unexercised",
    }
    stages.append(result)
    print(
        "REHEARSAL_STAGE_UNEXERCISED "
        f"stage={stage} blocked_by={','.join(blocked_by)}",
        file=sys.stderr,
        flush=True,
    )


def _write_stage_summary(
    *,
    evidence_root: Path,
    candidate_sha: str,
    elapsed_seconds: float,
    profile: str,
    stages: Sequence[dict[str, Any]],
) -> Path:
    failed = sum(item.get("status") == "failed" for item in stages)
    unexercised = sum(
        item.get("status") == "unexercised" for item in stages
    )
    output = evidence_root / (
        f"leadpoet-restart-rehearsal-{candidate_sha}-{profile}-stages.json"
    )
    output.write_text(
        json.dumps(
            {
                "candidate_sha": candidate_sha,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "failure_count": failed,
                "profile": profile,
                "schema_version": (
                    "leadpoet.local_restart_rehearsal_stage_summary.v1"
                ),
                "stage_count": len(stages),
                "stages": list(stages),
                "status": (
                    "passed" if failed == 0 and unexercised == 0 else "failed"
                ),
                "target_seconds": PROFILE_LIMITS[profile]["target_seconds"],
                "unexercised_count": unexercised,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


@contextmanager
def _recording_fixture_stack(
    stages: list[dict[str, Any]],
) -> Iterator[ExitStack]:
    stack = ExitStack()
    try:
        yield stack
    except KeyboardInterrupt:
        raise
    except RehearsalTimeBudgetExceeded:
        raise
    except BaseException as exc:
        result = _stage_result_from_exception(
            stage="fixture-orchestration",
            exc=exc,
        )
        stages.append(result)
        print(
            "REHEARSAL_STAGE_FAILED_CONTINUING "
            "stage=fixture-orchestration "
            f"error_type={result['error_type']} error={result['error']!r}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        _run_independent_stage(
            stage="fixture-cleanup",
            action=stack.close,
            stages=stages,
        )


@contextmanager
def _prepared_fixture_seed(
    tag: str,
    *,
    source_root: Path,
    candidate_sha: str,
    drand_artifact_root: Path,
    docker_platform: str,
    profile: str,
) -> Iterator[Path]:
    """Build immutable sanitized release fixtures once per target SHA."""

    limits = PROFILE_LIMITS[profile]
    with tempfile.TemporaryDirectory(
        prefix=f"leadpoet-rehearsal-fixture-{candidate_sha[:12]}-"
    ) as raw:
        root = Path(raw)
        generated_state = root / "generated-state"
        generated_config = root / "generated-config"
        seed = root / "seed"
        generated_state.mkdir()
        generated_config.mkdir()
        seed.mkdir()
        try:
            _run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--platform",
                    docker_platform,
                    "--network",
                    "none",
                    "--cpus",
                    str(limits["cpus"]),
                    "--memory",
                    str(limits["memory"]),
                    "--pids-limit",
                    "2048",
                    "--security-opt",
                    "no-new-privileges",
                    "--tmpfs",
                    "/tmp:rw,exec,nosuid,size=2g",
                    "--mount",
                    f"type=bind,src={source_root},dst=/source,readonly",
                    *_SOURCE_GIT_SAFE_DIRECTORY_DOCKER_ARGS,
                    "--mount",
                    (
                        f"type=bind,src={drand_artifact_root},"
                        "dst=/opt/leadpoet/drand-cabi-v2,readonly"
                    ),
                    "--mount",
                    (
                        f"type=bind,src={generated_state},"
                        "dst=/rehearsal-state"
                    ),
                    "--mount",
                    (
                        f"type=bind,src={generated_config},"
                        "dst=/fixture-config"
                    ),
                    "--env",
                    "REHEARSAL_COMPONENT=validator",
                    "--env",
                    f"REHEARSAL_CANDIDATE_SHA={candidate_sha}",
                    "--env",
                    "REHEARSAL_SCOPE=exact",
                    "--env",
                    "REHEARSAL_STATE_ROOT=/rehearsal-state",
                    "--env",
                    "PYTHONPATH=/source:/harness",
                    "--entrypoint",
                    "/usr/bin/python3.11",
                    tag,
                    "/harness/prepare_host_fixtures.py",
                    "--output-dir",
                    "/fixture-config",
                    "--candidate-sha",
                    candidate_sha,
                ],
                capture=True,
            )
        except BaseException as original:
            fixture_diagnostic = _report_fixture_generation_failure(original)
            try:
                setattr(
                    original,
                    "_rehearsal_fixture_generation_diagnostic",
                    fixture_diagnostic,
                )
            except BaseException:
                pass
            _normalize_evidence_after_failure(
                tag,
                evidence_root=root,
                docker_platform=docker_platform,
                original=original,
            )
            raise
        else:
            _normalize_evidence_ownership(
                tag,
                evidence_root=root,
                docker_platform=docker_platform,
            )
        release_input = generated_state / "release-build-input.json"
        validator_app = generated_state / "validator-app"
        gateway_identities = (
            generated_state / "gateway-enclave-build-identities"
        )
        gateway_attested_runtime = (
            generated_state / "gateway-attested-runtime"
        )
        if (
            not release_input.is_file()
            or not validator_app.is_dir()
            or not gateway_identities.is_dir()
            or not gateway_attested_runtime.is_dir()
        ):
            raise SystemExit("sanitized fixture seed is incomplete")
        release = json.loads(release_input.read_text(encoding="utf-8"))
        if release.get("commit_sha") != candidate_sha:
            raise SystemExit("sanitized fixture seed commit differs")
        shutil.copytree(generated_config, seed / "config-v2")
        shutil.copy2(release_input, seed / release_input.name)
        shutil.copytree(validator_app, seed / validator_app.name)
        shutil.copytree(
            gateway_identities,
            seed / gateway_identities.name,
        )
        shutil.copytree(
            gateway_attested_runtime,
            seed / gateway_attested_runtime.name,
        )
        (seed / "fixture-seed.json").write_text(
            json.dumps(
                {
                    "schema_version": "leadpoet.local_fixture_seed.v1",
                    "candidate_sha": candidate_sha,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        yield seed


def _run_workflow(
    tag: str,
    *,
    source_root: Path,
    evidence_root: Path,
    from_sha: str,
    candidate_sha: str,
    profile: str,
    docker_platform: str,
    production_allocation: Path | None = None,
) -> None:
    limits = PROFILE_LIMITS[profile]
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        docker_platform,
        "--network",
        "none",
        "--cpus",
        str(limits["cpus"]),
        "--memory",
        str(limits["memory"]),
        "--pids-limit",
        "2048",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=2g",
        "--mount",
        f"type=bind,src={source_root},dst=/source,readonly",
        *_SOURCE_GIT_SAFE_DIRECTORY_DOCKER_ARGS,
        "--mount",
        f"type=bind,src={evidence_root},dst=/evidence",
        "--env",
        "REHEARSAL_COMPONENT=workflow",
        "--env",
        f"REHEARSAL_FROM_SHA={from_sha}",
        "--env",
        f"REHEARSAL_CANDIDATE_SHA={candidate_sha}",
        "--env",
        "REHEARSAL_TRANSITION=forward",
        "--env",
        "REHEARSAL_SCOPE=exact",
        "--env",
        "PYTHONPATH=/source:/harness",
        "--env",
        f"REHEARSAL_PROFILE={profile}",
        "--env",
        f"REHEARSAL_EPOCHS={limits['epochs']}",
        "--env",
        "REHEARSAL_FAULT_MATRIX="
        + ("1" if limits["fault_matrix"] else "0"),
        "--env",
        "REHEARSAL_WEIGHT_READINESS_FAIL_ONCE=1",
        "--env",
        "GATEWAY_WEIGHT_INPUT_REPAIR_RETRY_SECONDS=0",
    ]
    if production_allocation is not None:
        command.extend(
            [
                "--mount",
                (
                    f"type=bind,src={production_allocation.resolve()},"
                    "dst=/rehearsal-production-allocation.json,readonly"
                ),
                "--env",
                (
                    "REHEARSAL_PRODUCTION_ALLOCATION="
                    "/rehearsal-production-allocation.json"
                ),
            ]
        )
    command.append(tag)
    try:
        _run(command)
    except (KeyboardInterrupt, RehearsalTimeBudgetExceeded):
        raise
    except BaseException:
        raise


def _join_evidence(
    tag: str,
    *,
    source_root: Path,
    evidence_root: Path,
    from_sha: str,
    candidate_sha: str,
    profile: str,
    docker_platform: str,
) -> Path:
    output = evidence_root / (
        f"leadpoet-restart-rehearsal-{candidate_sha}-{profile}.json"
    )
    command = [
            "docker",
            "run",
            "--rm",
            "--platform",
            docker_platform,
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "1g",
            "--pids-limit",
            "128",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=bind,src={source_root},dst=/source,readonly",
            *_SOURCE_GIT_SAFE_DIRECTORY_DOCKER_ARGS,
            "--mount",
            f"type=bind,src={evidence_root},dst=/evidence",
            "--entrypoint",
            "/usr/bin/python3.11",
            tag,
            "/harness/join_evidence.py",
            "--evidence-root",
            "/evidence",
            "--from-sha",
            from_sha,
            "--candidate-sha",
            candidate_sha,
            "--profile",
            profile,
            "--output",
            f"/evidence/{output.name}",
        ]
    try:
        _run(command)
        if not output.is_file():
            raise SystemExit(
                "joined restart rehearsal evidence was not produced"
            )
    except BaseException:
        raise
    return output


def _run_python37_finalization_probe(source_root: Path) -> None:
    """Exercise the measured enclave's post-broadcast path under CPython 3.7."""

    _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "512m",
            "--pids-limit",
            "128",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=bind,src={source_root},dst=/source,readonly",
            *_SOURCE_GIT_SAFE_DIRECTORY_DOCKER_ARGS,
            "--env",
            "PYTHONPATH=/source",
            "--workdir",
            "/source",
            PYTHON37_IMAGE,
            "python",
            "tests/validator_enclave_python37_runtime_probe.py",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == [_DRAND_INTERNAL_DOWNLOAD_COMMAND]:
        return _download_locked_drand_source(raw_argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-sha",
        default="HEAD^",
        help="Currently deployed N-1 commit whose installed launcher starts the test.",
    )
    parser.add_argument("--candidate-sha", default="HEAD")
    parser.add_argument(
        "--transition",
        choices=("auto", "forward", "rollback"),
        default="auto",
    )
    parser.add_argument(
        "--profile",
        choices=CLI_PROFILES,
        default="prepush",
        help=(
            "prepush is the default 5-10 minute gate; unaccelerated runs "
            "forward/rollback/roll-forward, the full fault matrix, and 100 epochs"
        ),
    )
    parser.add_argument(
        "--gateway-worker-fleet-mode",
        choices=("active", "deferred"),
        default="active",
        help=(
            "active exercises compliant TLS proxy workers; deferred exercises "
            "the explicit one-restart recovery path while retaining the same "
            "compliant TLS proxy validation"
        ),
    )
    parser.add_argument("--rebuild-image", action="store_true")
    parser.add_argument(
        "--production-allocation",
        type=Path,
        help=(
            "Optional hash-bound allocation emitted by a disposable "
            "production-parity gateway. It is mounted read-only into only "
            "the canonical workflow stage."
        ),
    )
    args = parser.parse_args(raw_argv)
    if args.production_allocation is not None:
        args.production_allocation = args.production_allocation.resolve()
        if not args.production_allocation.is_file():
            parser.error("--production-allocation must be a readable file")
    args.profile = _runtime_profile(args.profile)

    with (
        _exclusive_rehearsal_lock(),
        _isolated_docker_client_config() as docker_client_root,
    ):
        target_seconds = PROFILE_LIMITS[args.profile]["target_seconds"]
        try:
            with _profile_time_limit(target_seconds):
                return _run_profile(
                    args,
                    docker_client_root=docker_client_root,
                )
        except RehearsalTimeBudgetExceeded as exc:
            print(
                "REHEARSAL_TIME_BUDGET_EXCEEDED "
                f"profile={args.profile} error={str(exc)!r}",
                file=sys.stderr,
                flush=True,
            )
            return 1


def _run_profile(
    args: argparse.Namespace,
    *,
    docker_client_root: Path,
) -> int:
    profile_started = time.monotonic()
    target_seconds = PROFILE_LIMITS[args.profile]["target_seconds"]
    print(
        "REHEARSAL_TIME_BUDGET "
        f"profile={args.profile} target_seconds={target_seconds or 'unbounded'}",
        flush=True,
    )
    from_sha = _git_sha(args.from_sha)
    candidate_sha = _git_sha(args.candidate_sha)
    transition = _resolve_transition(
        from_sha,
        candidate_sha,
        args.transition,
    )
    if transition != "forward":
        raise SystemExit(
            "profile rehearsals must start with the deployed N-1 commit and "
            "a descendant candidate; the unaccelerated profile performs its own "
            "rollback and roll-forward"
        )
    harness_sha = candidate_sha
    _verify_driver_identity(harness_sha)

    docker_platform = _docker_platform(args.profile)
    buildx_executable = _provision_official_buildx(docker_client_root)
    wheelhouse_shas = (from_sha, candidate_sha)
    tag = _image_tag(
        harness_sha,
        docker_platform=docker_platform,
        wheelhouse_shas=wheelhouse_shas,
    )
    image_build_required = args.rebuild_image or not _image_exists(tag)

    def build_image() -> None:
        _build_image(
            tag,
            harness_sha=harness_sha,
            docker_platform=docker_platform,
            buildx_executable=buildx_executable,
            wheelhouse_shas=wheelhouse_shas,
        )
    if args.profile != "prepush" and image_build_required:
        build_image()

    overlap_source_snapshot = (
        args.profile == "prepush" and image_build_required
    )
    if overlap_source_snapshot:
        source_snapshot_context = _isolated_source_snapshot_root()
    else:
        source_snapshot_context = _isolated_source_snapshot(
            harness_sha=harness_sha,
            required_shas=(from_sha, candidate_sha),
        )

    with source_snapshot_context as source_root:
        stage_results: list[dict[str, Any]] = []
        target_shas = tuple(sorted({from_sha, candidate_sha}))
        drand_artifacts: dict[str, Path] = {}
        # Cold runs defer this probe behind the pre-scheduler drand preparation,
        # but the candidate-bound inventory retains its established position.
        python37_stage_index = len(stage_results)
        print(
            "Running validator enclave finalization proof under CPython 3.7",
            flush=True,
        )

        def run_python37_probe() -> None:
            _run_independent_stage(
                stage="python37-finalization",
                action=lambda: _run_python37_finalization_probe(source_root),
                stages=stage_results,
            )

        if overlap_source_snapshot:
            cold_drand_results: list[dict[str, Any]] = []

            def prepare_cold_drand_artifacts() -> None:
                for target in target_shas:
                    stage = f"drand-artifact-{target[:12]}"
                    passed, artifact = _run_independent_stage(
                        stage=stage,
                        action=lambda target=target: _prepare_drand_artifact(
                            source_root=source_root,
                            candidate_sha=target,
                            buildx_executable=buildx_executable,
                        ),
                        stages=cold_drand_results,
                    )
                    if passed:
                        drand_artifacts[target] = artifact

            _run_prepush_image_snapshot_and_artifacts(
                image_action=build_image,
                snapshot_action=lambda: _populate_isolated_source_snapshot(
                    source=source_root,
                    harness_sha=harness_sha,
                    required_shas=(from_sha, candidate_sha),
                ),
                artifact_action=prepare_cold_drand_artifacts,
            )
            stage_results.extend(cold_drand_results)
        else:
            run_python37_probe()

        evidence_handoff_attempted = False
        with _temporary_evidence_directory(
            tag,
            docker_platform=docker_platform,
            handoff_attempted=lambda: evidence_handoff_attempted,
        ) as evidence_root:
            transitions = (transition,)
            if args.profile == "release" and transition == "forward":
                transitions = ("forward", "rollback", "forward")
            workflow_stage = f"workflow-{args.profile}"
            workflow_action = (
                workflow_stage,
                partial(
                    _run_workflow,
                    tag,
                    source_root=source_root,
                    from_sha=from_sha,
                    candidate_sha=candidate_sha,
                    evidence_root=evidence_root,
                    profile=args.profile,
                    docker_platform=docker_platform,
                    production_allocation=args.production_allocation,
                ),
            )
            deferred_workflow_results: list[dict[str, Any]] | None = None
            # Candidate startup can reconstruct receipts issued by the
            # deployed release, so both immutable release channels must be
            # available even in the one-way prepush profile.
            with _recording_fixture_stack(stage_results) as fixture_stack:
                fixture_seeds: dict[str, Path] = {}
                if not overlap_source_snapshot:
                    for target in target_shas:
                        stage = f"drand-artifact-{target[:12]}"
                        passed, artifact = _run_independent_stage(
                            stage=stage,
                            action=lambda target=target: _prepare_drand_artifact(
                                source_root=source_root,
                                candidate_sha=target,
                                buildx_executable=buildx_executable,
                            ),
                            stages=stage_results,
                        )
                        if passed:
                            drand_artifacts[target] = artifact
                durable_state_root = evidence_root / "durable-boundary-state"
                durable_state_root.mkdir(mode=0o700)

                def prepare_fixture_seed(target: str) -> None:
                    stage = f"fixture-seed-{target[:12]}"
                    dependency = f"drand-artifact-{target[:12]}"
                    if target not in drand_artifacts:
                        _mark_stage_unexercised(
                            stage=stage,
                            blocked_by=[dependency],
                            stages=stage_results,
                        )
                        return
                    passed, seed = _run_independent_stage(
                        stage=stage,
                        action=lambda target=target: fixture_stack.enter_context(
                            _prepared_fixture_seed(
                                tag,
                                source_root=source_root,
                                candidate_sha=target,
                                drand_artifact_root=drand_artifacts[target],
                                docker_platform=docker_platform,
                                profile=args.profile,
                            )
                        ),
                        stages=stage_results,
                    )
                    if passed:
                        fixture_seeds[target] = seed

                def component_actions_for_transition(
                    ordinal: int,
                    run_transition: str,
                ) -> list[tuple[str, Callable[[], Any]]]:
                    run_from = from_sha
                    run_candidate = candidate_sha
                    if run_transition == "rollback":
                        run_from, run_candidate = candidate_sha, from_sha
                    elif ordinal == 2:
                        run_from, run_candidate = from_sha, candidate_sha
                    component_actions: list[
                        tuple[str, Callable[[], Any]]
                    ] = []
                    for component in ("gateway", "validator"):
                        print(
                            f"Running isolated {component} restart rehearsal "
                            f"{run_from[:12]} -> {run_candidate[:12]} "
                            f"transition={run_transition} profile={args.profile}",
                            flush=True,
                        )
                        stage = (
                            f"{component}-{run_transition}-{ordinal + 1}"
                        )
                        blocked_by = []
                        if run_candidate not in drand_artifacts:
                            blocked_by.append(
                                f"drand-artifact-{run_candidate[:12]}"
                            )
                        if run_candidate not in fixture_seeds:
                            blocked_by.append(
                                f"fixture-seed-{run_candidate[:12]}"
                            )
                        if run_from not in fixture_seeds:
                            blocked_by.append(
                                f"fixture-seed-{run_from[:12]}"
                            )
                        if (
                            candidate_sha != run_candidate
                            and candidate_sha not in fixture_seeds
                        ):
                            blocked_by.append(
                                f"fixture-seed-{candidate_sha[:12]}"
                            )
                        if blocked_by:
                            _mark_stage_unexercised(
                                stage=stage,
                                blocked_by=blocked_by,
                                stages=stage_results,
                            )
                            continue
                        component_actions.append(
                            (
                                stage,
                                partial(
                                    _run_component,
                                    tag,
                                    source_root=source_root,
                                    component=component,
                                    from_sha=run_from,
                                    candidate_sha=run_candidate,
                                    transition=run_transition,
                                    evidence_root=evidence_root,
                                    drand_artifact_root=drand_artifacts[
                                        run_candidate
                                    ],
                                    profile=args.profile,
                                    docker_platform=docker_platform,
                                    fixture_seed_root=fixture_seeds[
                                        run_candidate
                                    ],
                                    from_fixture_seed_root=fixture_seeds[
                                        run_from
                                    ],
                                    durable_fixture_seed_root=fixture_seeds[
                                        candidate_sha
                                    ],
                                    durable_state_root=durable_state_root,
                                    durable_schema_sha=candidate_sha,
                                    run_ordinal=ordinal + 1,
                                    gateway_worker_fleet_mode=(
                                        args.gateway_worker_fleet_mode
                                    ),
                                ),
                            )
                        )
                    return component_actions

                fixture_targets = sorted(target_shas)
                if args.profile == "prepush":

                    def prepare_prepush_components() -> Sequence[
                        tuple[str, Callable[[], Any]]
                    ]:
                        for target in fixture_targets:
                            prepare_fixture_seed(target)
                        return component_actions_for_transition(0, transition)

                    print(
                        "Running production V2 workflow alongside fixture "
                        "preparation (prepush)",
                        flush=True,
                    )
                    deferred_workflow_results = _run_prepush_runtime_stages(
                        preparation_action=prepare_prepush_components,
                        workflow_action=workflow_action,
                        expected_component_stages=(
                            f"gateway-{transition}-1",
                            f"validator-{transition}-1",
                        ),
                        stages=stage_results,
                        worker_prefix_action=(
                            (
                                "python37-finalization",
                                lambda: _run_python37_finalization_probe(
                                    source_root
                                ),
                            )
                            if overlap_source_snapshot
                            else None
                        ),
                        worker_prefix_stage_index=(
                            python37_stage_index
                            if overlap_source_snapshot
                            else None
                        ),
                    )
                else:
                    for target in fixture_targets:
                        prepare_fixture_seed(target)
                    for ordinal, run_transition in enumerate(transitions):
                        for stage, action in component_actions_for_transition(
                            ordinal,
                            run_transition,
                        ):
                            _run_independent_stage(
                                stage=stage,
                                action=action,
                                stages=stage_results,
                            )
            if deferred_workflow_results is None:
                print(
                    "Running production V2 workflow against strict local "
                    f"boundaries ({args.profile})",
                    flush=True,
                )
                _run_independent_stage(
                    stage=workflow_action[0],
                    action=workflow_action[1],
                    stages=stage_results,
                )
            else:
                stage_results.extend(deferred_workflow_results)
            required_join_stages = [
                item["stage"]
                for item in stage_results
                if (
                    item["stage"].startswith(("gateway-", "validator-"))
                    or item["stage"] == workflow_stage
                )
            ]
            stage_status = {
                item["stage"]: item["status"] for item in stage_results
            }
            evidence: Path | None = None
            join_stage = f"evidence-join-{args.profile}"
            blocked_join_stages = [
                stage
                for stage in required_join_stages
                if stage_status.get(stage) != "passed"
            ]
            if blocked_join_stages:
                _mark_stage_unexercised(
                    stage=join_stage,
                    blocked_by=blocked_join_stages,
                    stages=stage_results,
                )
            else:
                _, evidence = _run_independent_stage(
                    stage=join_stage,
                    action=lambda: _join_evidence(
                        tag,
                        source_root=source_root,
                        evidence_root=evidence_root,
                        from_sha=from_sha,
                        candidate_sha=candidate_sha,
                        profile=args.profile,
                        docker_platform=docker_platform,
                    ),
                    stages=stage_results,
                )
            # All workflow, component, and join writers are terminal. This is
            # the sole authoritative handoff before any shared-tree projection,
            # durable copy, or joined-evidence read.
            evidence_handoff_attempted = True
            _complete_shared_evidence_handoff(
                tag,
                stages=stage_results,
                workflow_stage=workflow_stage,
                evidence_root=evidence_root,
                candidate_sha=candidate_sha,
                profile=args.profile,
                docker_platform=docker_platform,
            )
            elapsed_seconds = time.monotonic() - profile_started
            if target_seconds is not None:
                budget_result = {
                    "duration_seconds": round(elapsed_seconds, 3),
                    "stage": "time-budget",
                    "status": (
                        "passed"
                        if elapsed_seconds <= target_seconds
                        else "failed"
                    ),
                    "target_seconds": target_seconds,
                }
                if elapsed_seconds > target_seconds:
                    budget_result["error"] = (
                        "prepush rehearsal exceeded its 10-minute budget"
                    )
                    budget_result["error_type"] = "RehearsalTimeBudgetExceeded"
                stage_results.append(budget_result)
                print(
                    "REHEARSAL_TIME_BUDGET_RESULT "
                    f"status={budget_result['status']} "
                    f"elapsed_seconds={budget_result['duration_seconds']} "
                    f"target_seconds={target_seconds}",
                    flush=True,
                )
            stage_summary = _write_stage_summary(
                evidence_root=evidence_root,
                candidate_sha=candidate_sha,
                elapsed_seconds=elapsed_seconds,
                profile=args.profile,
                stages=stage_results,
            )
            incomplete = [
                item
                for item in stage_results
                if item.get("status") != "passed"
            ]
            if incomplete:
                _emit_terminal_stage_diagnostics(stage_results)
                durable_failure = _preserve_batched_failure_evidence(
                    evidence_root=evidence_root,
                    candidate_sha=candidate_sha,
                    stages=stage_results,
                )
                raise SystemExit(
                    f"{args.profile} rehearsal failed after completing independent "
                    f"stages; evidence={durable_failure}"
                )
            if evidence is None:
                raise SystemExit("joined rehearsal evidence is unexpectedly absent")
            durable_output = Path(tempfile.gettempdir()) / evidence.name
            durable_output.write_bytes(evidence.read_bytes())
            durable_stage_output = (
                Path(tempfile.gettempdir()) / stage_summary.name
            )
            durable_stage_output.write_bytes(stage_summary.read_bytes())
            print(f"REHEARSAL_EVIDENCE {durable_output}", flush=True)
            print(
                f"REHEARSAL_STAGE_EVIDENCE {durable_stage_output}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
