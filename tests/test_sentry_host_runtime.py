from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from leadpoet_observability import host_runtime


PINNED = "2.66.1"


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(f"sentry-sdk=={PINNED}\n", encoding="utf-8")
    lock = tmp_path / "requirements-host.lock"
    lock.write_text(
        f"sentry-sdk=={PINNED} \\\n    --hash=sha256:{'a' * 64}\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    return requirements, lock, cache


def test_pinned_requirement_and_lock_must_match(tmp_path):
    requirements, lock, cache = _inputs(tmp_path)
    lock.write_text(
        "sentry-sdk==2.66.0 \\\n    --hash=sha256:" + "b" * 64 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(host_runtime.HostRuntimeError, match="version_mismatch"):
        host_runtime.prepare_host_runtime(
            base_python=Path(sys.executable),
            repo_root=Path.cwd(),
            requirements_path=requirements,
            lock_path=lock,
            cache_root=cache,
        )


def test_existing_authoritative_interpreter_is_reused_without_build(
    monkeypatch, tmp_path
):
    requirements, lock, cache = _inputs(tmp_path)
    base = Path(sys.executable)
    monkeypatch.setattr(host_runtime, "_runtime_works", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        host_runtime,
        "_create_runtime",
        lambda *args, **kwargs: pytest.fail("runtime build was not expected"),
    )

    result = host_runtime.prepare_host_runtime(
        base_python=base,
        repo_root=Path.cwd(),
        requirements_path=requirements,
        lock_path=lock,
        cache_root=cache,
    )

    assert result == base
    assert not cache.exists()


def test_runtime_is_built_atomically_then_reused(monkeypatch, tmp_path):
    requirements, lock, cache = _inputs(tmp_path)
    base = Path(sys.executable)
    builds = []

    def runtime_works(python_bin, **kwargs):
        return python_bin != base and python_bin.is_file() and os.access(python_bin, os.X_OK)

    def create_runtime(destination, **kwargs):
        builds.append(destination)
        python_bin = destination / "bin" / "python"
        python_bin.parent.mkdir(parents=True)
        python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python_bin.chmod(0o700)

    monkeypatch.setattr(host_runtime, "_runtime_works", runtime_works)
    monkeypatch.setattr(host_runtime, "_create_runtime", create_runtime)

    first = host_runtime.prepare_host_runtime(
        base_python=base,
        repo_root=Path.cwd(),
        requirements_path=requirements,
        lock_path=lock,
        cache_root=cache,
    )
    second = host_runtime.prepare_host_runtime(
        base_python=base,
        repo_root=Path.cwd(),
        requirements_path=requirements,
        lock_path=lock,
        cache_root=cache,
    )

    assert first == second
    assert first.is_file()
    assert len(builds) == 1
    assert not (cache / ".runtime-staging").exists()


def test_failed_build_is_cleaned_and_never_published(monkeypatch, tmp_path):
    requirements, lock, cache = _inputs(tmp_path)
    base = Path(sys.executable)
    monkeypatch.setattr(host_runtime, "_runtime_works", lambda *args, **kwargs: False)

    def fail_build(*args, **kwargs):
        raise host_runtime.HostRuntimeError("dependency_install", "TimeoutExpired")

    monkeypatch.setattr(host_runtime, "_create_runtime", fail_build)

    with pytest.raises(host_runtime.HostRuntimeError, match="TimeoutExpired"):
        host_runtime.prepare_host_runtime(
            base_python=base,
            repo_root=Path.cwd(),
            requirements_path=requirements,
            lock_path=lock,
            cache_root=cache,
        )

    assert not (cache / ".runtime-staging").exists()
    assert [item.name for item in cache.iterdir()] == [".prepare.lock"]


def test_cli_failure_is_sanitized_and_nonzero(tmp_path, capsys):
    requirements, lock, cache = _inputs(tmp_path)
    requirements.write_text("sentry-sdk>=2\n", encoding="utf-8")

    result = host_runtime.main(
        [
            "--base-python",
            sys.executable,
            "--repo-root",
            str(Path.cwd()),
            "--requirements",
            str(requirements),
            "--lock",
            str(lock),
            "--cache-root",
            str(cache),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "telemetry.host_runtime_unavailable" in captured.err
    assert "pinned_requirement_invalid" in captured.err
    assert "sentry-sdk>=2" not in captured.err
