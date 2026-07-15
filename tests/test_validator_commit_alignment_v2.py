from __future__ import annotations

import pytest

from validator_tee.host.commit_alignment_v2 import (
    ValidatorCommitAlignmentV2Error,
    verify_validator_v2_commit_alignment,
)


COMMIT = "1" * 40


class _Client:
    def __init__(self, commit: str = COMMIT):
        self.commit = commit

    def get_authoritative_v2_boot_identity(self):
        return {
            "commit_sha": self.commit,
            "boot_identity_hash": "sha256:" + "2" * 64,
            "pcr0": "3" * 96,
        }


def test_validator_host_and_enclave_must_use_exact_restart_commit(monkeypatch):
    result = verify_validator_v2_commit_alignment(
        _Client(),
        expected_commit=COMMIT,
        host_commit=COMMIT,
        required=True,
    )
    assert result["commit_sha"] == COMMIT
    assert result["pcr0"] == "3" * 96


def test_validator_rejects_host_commit_drift(monkeypatch):
    with pytest.raises(ValidatorCommitAlignmentV2Error, match="host commit differs"):
        verify_validator_v2_commit_alignment(
            _Client(),
            expected_commit=COMMIT,
            host_commit="4" * 40,
            required=True,
        )


def test_validator_rejects_enclave_commit_drift(monkeypatch):
    with pytest.raises(
        ValidatorCommitAlignmentV2Error, match="enclave commit differs"
    ):
        verify_validator_v2_commit_alignment(
            _Client("5" * 40),
            expected_commit=COMMIT,
            host_commit=COMMIT,
            required=True,
        )


def test_validator_production_requires_restart_commit(monkeypatch):
    with pytest.raises(
        ValidatorCommitAlignmentV2Error, match="required in production"
    ):
        verify_validator_v2_commit_alignment(
            _Client(), expected_commit="", host_commit=COMMIT, required=True
        )
