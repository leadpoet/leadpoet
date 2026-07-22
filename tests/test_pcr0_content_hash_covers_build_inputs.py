"""The PCR0 cache key must cover every validator EIF build input.

A cache key derived from a hand-maintained subset of files let two builds
with different enclave code collide on one key: the gateway skipped the
rebuild, kept serving a stale PCR0 as HEAD's, and rejected the actually
deployed validator with "validator PCR0 is absent from the dynamic Git
build cache". Every path in PCR0_COPY_PATHS — including protected workflow
manifests and non .py/.json/.txt artifacts such as the drand lock and
sha256 pins — must change the computed content hash.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gateway.utils.pcr0_builder import (
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
    """Copy only the PCR0 build inputs into a scratch repo directory."""

    destination = tmp_path / "repo"
    for entry in PCR0_COPY_PATHS:
        source = REPO_ROOT / entry
        target = destination / entry
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return destination


def test_every_sensitive_build_input_exists_and_is_copied(repo_copy):
    for entry in SENSITIVE_BUILD_INPUTS:
        assert (repo_copy / entry).is_file(), entry


def test_content_hash_changes_when_any_build_input_changes(repo_copy):
    baseline = compute_files_content_hash(str(repo_copy))
    assert baseline is not None

    for entry in SENSITIVE_BUILD_INPUTS:
        target = repo_copy / entry
        original = target.read_bytes()
        target.write_bytes(original + b"\n# pcr0-key-coverage-probe\n")
        try:
            mutated = compute_files_content_hash(str(repo_copy))
        finally:
            target.write_bytes(original)
        assert mutated is not None
        assert mutated != baseline, (
            f"{entry} does not affect the PCR0 cache key; a stale PCR0 "
            "would be served for builds that change it"
        )

    assert compute_files_content_hash(str(repo_copy)) == baseline
