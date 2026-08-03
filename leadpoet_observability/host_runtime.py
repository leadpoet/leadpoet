"""Prepare a bounded host-only Python runtime for restart telemetry.

The validator's authoritative interpreter is deliberately never changed. This
module creates a small, hash-locked, cached environment used only by the
best-effort Sentry restart summary emitted from the parent host.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence


_SENTRY_REQUIREMENT_RE = re.compile(
    r"^sentry-sdk==(?P<version>[A-Za-z0-9][A-Za-z0-9_.+-]*)$",
    re.IGNORECASE,
)
_DEFAULT_PREPARE_TIMEOUT_SECONDS = 30.0
_LOCK_WAIT_SECONDS = 1.0


class HostRuntimeError(RuntimeError):
    """A sanitized, non-authoritative telemetry-runtime preparation failure."""

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(f"{stage}:{reason}")
        self.stage = stage
        self.reason = reason


def _pinned_sentry_version(requirements_path: Path) -> str:
    matches = []
    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HostRuntimeError("requirements", type(exc).__name__) from None
    for raw in lines:
        candidate = raw.split("#", 1)[0].strip()
        match = _SENTRY_REQUIREMENT_RE.fullmatch(candidate)
        if match:
            matches.append(match.group("version"))
    if len(matches) != 1:
        raise HostRuntimeError("requirements", "pinned_requirement_invalid")
    return matches[0]


def _lock_sentry_version(lock_path: Path) -> str:
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HostRuntimeError("lockfile", type(exc).__name__) from None
    matches = []
    for raw in lines:
        candidate = raw.split("#", 1)[0].strip().rstrip("\\").strip()
        match = _SENTRY_REQUIREMENT_RE.fullmatch(candidate)
        if match:
            matches.append(match.group("version"))
    if len(matches) != 1:
        raise HostRuntimeError("lockfile", "pinned_requirement_invalid")
    return matches[0]


def _runtime_key(lock_path: Path, base_python: Path) -> str:
    try:
        lock_bytes = lock_path.read_bytes()
        python_identity = subprocess.run(
            [
                str(base_python),
                "-c",
                "import sys; print(sys.implementation.name); print(sys.version_info[:2])",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.encode("utf-8")
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostRuntimeError("python_identity", type(exc).__name__) from None
    return hashlib.sha256(lock_bytes + b"\0" + python_identity).hexdigest()[:24]


def _validation_environment(repo_root: Path) -> Mapping[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("SENTRY_") or key.startswith("LEADPOET_SENTRY_"):
            environment.pop(key, None)
    environment.update(
        {
            "PYTHONPATH": str(repo_root),
            "LEADPOET_SENTRY_ENABLED": "1",
            "LEADPOET_SENTRY_DSN": (
                "https://00000000000000000000000000000000@example.invalid/1"
            ),
            "LEADPOET_SENTRY_ENVIRONMENT": "host-runtime-preflight",
            "LEADPOET_SENTRY_RELEASE": "0" * 40,
            "LEADPOET_SENTRY_TRACES_SAMPLE_RATE": "0",
            "LEADPOET_SENTRY_MESSAGE_MODE": "redact-all",
        }
    )
    return environment


def _runtime_works(
    python_bin: Path,
    *,
    repo_root: Path,
    expected_version: str,
) -> bool:
    if not python_bin.exists() or not os.access(python_bin, os.X_OK):
        return False
    probe = (
        "from importlib.metadata import version; "
        f"assert version('sentry-sdk') == {expected_version!r}; "
        "from leadpoet_observability.sentry_bootstrap import init_sentry; "
        "raise SystemExit(0 if init_sentry('validator-host-runtime-preflight') else 1)"
    )
    try:
        completed = subprocess.run(
            [str(python_bin), "-c", probe],
            check=False,
            env=_validation_environment(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=4,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _pip_environment() -> Mapping[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("PIP_"):
            environment.pop(key, None)
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PIP_INDEX_URL": "https://pypi.org/simple",
        }
    )
    return environment


def _run_build_step(
    command: Sequence[str],
    *,
    stage: str,
    timeout_seconds: float,
    environment: Optional[Mapping[str, str]] = None,
) -> None:
    try:
        subprocess.run(
            list(command),
            check=True,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(1.0, timeout_seconds),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostRuntimeError(stage, type(exc).__name__) from None


def _create_runtime(
    destination: Path,
    *,
    base_python: Path,
    lock_path: Path,
    timeout_seconds: float,
) -> None:
    started = time.monotonic()
    _run_build_step(
        [str(base_python), "-m", "venv", str(destination)],
        stage="venv_create",
        timeout_seconds=min(10.0, timeout_seconds),
    )
    remaining = timeout_seconds - (time.monotonic() - started)
    if remaining < 1.0:
        raise HostRuntimeError("dependency_install", "deadline_exhausted")
    _run_build_step(
        [
            str(destination / "bin" / "python"),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--require-hashes",
            "--requirement",
            str(lock_path),
        ],
        stage="dependency_install",
        timeout_seconds=remaining,
        environment=_pip_environment(),
    )


def _safe_cache_root(cache_root: Path) -> Path:
    cache_root = cache_root.expanduser()
    if cache_root.is_symlink():
        raise HostRuntimeError("cache", "symlink_refused")
    cache_root = cache_root.absolute()
    try:
        cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        cache_root.chmod(0o700)
    except OSError as exc:
        raise HostRuntimeError("cache", type(exc).__name__) from None
    return cache_root


def prepare_host_runtime(
    *,
    base_python: Path,
    repo_root: Path,
    requirements_path: Path,
    lock_path: Path,
    cache_root: Path,
    timeout_seconds: float = _DEFAULT_PREPARE_TIMEOUT_SECONDS,
) -> Path:
    """Return an SDK-capable interpreter without changing validator authority."""

    expected_version = _pinned_sentry_version(requirements_path)
    if _lock_sentry_version(lock_path) != expected_version:
        raise HostRuntimeError("lockfile", "version_mismatch")
    if _runtime_works(
        base_python,
        repo_root=repo_root,
        expected_version=expected_version,
    ):
        return base_python

    cache_root = _safe_cache_root(cache_root)
    runtime_key = _runtime_key(lock_path, base_python)
    destination = cache_root / runtime_key
    if destination.is_symlink():
        raise HostRuntimeError("cache", "symlink_refused")
    destination_python = destination / "bin" / "python"
    if _runtime_works(
        destination_python,
        repo_root=repo_root,
        expected_version=expected_version,
    ):
        return destination_python

    lock_file = cache_root / ".prepare.lock"
    try:
        descriptor = os.open(
            lock_file,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise HostRuntimeError("cache_lock", type(exc).__name__) from None
    try:
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if _runtime_works(
                    destination_python,
                    repo_root=repo_root,
                    expected_version=expected_version,
                ):
                    return destination_python
                if time.monotonic() >= deadline:
                    raise HostRuntimeError("cache_lock", "busy") from None
                time.sleep(0.05)

        if _runtime_works(
            destination_python,
            repo_root=repo_root,
            expected_version=expected_version,
        ):
            return destination_python

        temporary = cache_root / ".runtime-staging"
        if temporary.is_symlink():
            raise HostRuntimeError("cache", "symlink_refused")
        if temporary.exists():
            shutil.rmtree(temporary)
        old_umask = os.umask(0o077)
        try:
            _create_runtime(
                temporary,
                base_python=base_python,
                lock_path=lock_path,
                timeout_seconds=max(1.0, timeout_seconds),
            )
        finally:
            os.umask(old_umask)
        try:
            temporary_python = temporary / "bin" / "python"
            if not _runtime_works(
                temporary_python,
                repo_root=repo_root,
                expected_version=expected_version,
            ):
                raise HostRuntimeError("runtime_validation", "failed")
            if destination.exists():
                if destination.is_symlink():
                    raise HostRuntimeError("cache", "symlink_refused")
                shutil.rmtree(destination)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass

    if not _runtime_works(
        destination_python,
        repo_root=repo_root,
        expected_version=expected_version,
    ):
        raise HostRuntimeError("runtime_validation", "post_publish_failed")
    return destination_python


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_DEFAULT_PREPARE_TIMEOUT_SECONDS,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        python_bin = prepare_host_runtime(
            base_python=args.base_python,
            repo_root=args.repo_root,
            requirements_path=args.requirements,
            lock_path=args.lock,
            cache_root=args.cache_root,
            timeout_seconds=min(45.0, max(1.0, args.timeout_seconds)),
        )
    except HostRuntimeError as exc:
        print(
            "validator_sentry_host_runtime_unavailable "
            "failure_code=telemetry.host_runtime_unavailable "
            f"stage={exc.stage} reason={exc.reason}",
            file=sys.stderr,
        )
        return 1
    print(str(python_bin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
