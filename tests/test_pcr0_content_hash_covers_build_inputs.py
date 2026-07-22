"""PCR0 cache keys cover every exact validator EIF build input."""

from __future__ import annotations

import builtins
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import gateway.utils.pcr0_builder as pcr0_builder
from gateway.utils.pcr0_builder import (
    MONITORED_DIRS,
    MONITORED_FILES,
    PCR0_COPY_PATHS,
    compute_files_content_hash,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_BUILD_INPUTS = (
    "validator_tee/enclave/protected_workflows_v2.json",
    "validator_tee/enclave/weight_authority_v2.py",
    "validator_tee/enclave/chain_source_v2.py",
    "validator_tee/enclave/Cargo.drand-cabi-v2.lock",
    "validator_tee/enclave/libbittensor_drand_v2.sha256",
)


@pytest.fixture()
def repo_copy(tmp_path):
    destination = tmp_path / "repo"
    entries = sorted(
        set(MONITORED_FILES)
        | {path.rstrip("/") for path in MONITORED_DIRS}
        | {path.rstrip("/") for path in PCR0_COPY_PATHS}
    )
    for entry in entries:
        source = REPO_ROOT / entry
        target = destination / entry
        assert source.exists(), entry
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return destination


def test_content_hash_changes_when_any_sensitive_build_input_changes(repo_copy):
    baseline = compute_files_content_hash(str(repo_copy))
    assert baseline is not None

    for entry in SENSITIVE_BUILD_INPUTS:
        target = repo_copy / entry
        original = target.read_bytes()
        target.write_bytes(original + b"\n# pcr0-key-coverage-probe\n")
        try:
            assert compute_files_content_hash(str(repo_copy)) != baseline, entry
        finally:
            target.write_bytes(original)

    assert compute_files_content_hash(str(repo_copy)) == baseline


def test_content_hash_fails_closed_when_required_input_is_missing(repo_copy):
    (repo_copy / "validator_tee/runtime-artifacts-v2.lock.json").unlink()
    assert compute_files_content_hash(str(repo_copy)) is None


def test_content_hash_fails_closed_when_required_input_is_unreadable(
    repo_copy,
    monkeypatch,
):
    denied_path = os.path.abspath(
        repo_copy / "validator_tee/enclave/weight_authority_v2.py"
    )
    real_open = builtins.open

    def deny_one_input(path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) == denied_path:
            raise PermissionError("test-denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_one_input)
    assert compute_files_content_hash(str(repo_copy)) is None


@pytest.mark.asyncio
async def test_unreadable_input_aborts_builder_without_docker_or_cache_relabel(
    repo_copy,
    monkeypatch,
):
    existing_cache = {
        "old-content:old-base": {
            "pcr0": "old-pcr0",
            "commit_hash": "a" * 40,
        }
    }
    monkeypatch.setattr(pcr0_builder, "BUILD_DIR", str(repo_copy))
    monkeypatch.setattr(
        pcr0_builder,
        "clone_or_update_repo",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(pcr0_builder, "compute_files_content_hash", lambda _repo: None)
    prepare_base = AsyncMock(return_value=True)
    build_enclave = AsyncMock(return_value="new-pcr0")
    monkeypatch.setattr(pcr0_builder, "ensure_base_image_exists", prepare_base)
    monkeypatch.setattr(
        pcr0_builder,
        "build_enclave_and_extract_pcr0",
        build_enclave,
    )
    monkeypatch.setattr(pcr0_builder, "_pcr0_cache", existing_cache)
    monkeypatch.setattr(pcr0_builder, "_last_content_hash", "old-content")
    monkeypatch.setattr(pcr0_builder, "_build_in_progress", False)

    await pcr0_builder.check_and_build_pcr0()

    prepare_base.assert_not_awaited()
    build_enclave.assert_not_awaited()
    assert pcr0_builder._pcr0_cache == existing_cache
    assert pcr0_builder._last_content_hash == "old-content"
    assert pcr0_builder._build_in_progress is False
