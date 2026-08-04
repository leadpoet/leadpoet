from datetime import datetime, timedelta, timezone

import pytest

import gateway.tee.artifact_vault_v2 as vault_module
from gateway.tee.artifact_vault_v2 import (
    ArtifactVaultV2Error,
    EncryptedArtifactVaultV2,
)


BOOT_HASH = "sha256:" + "b" * 64
START = datetime(2026, 7, 27, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now


def _vault(clock):
    return EncryptedArtifactVaultV2(
        master_key=b"k" * 32,
        boot_identity_hash=BOOT_HASH,
        clock=clock,
    )


def _seal(vault, suffix):
    return vault.seal(
        ("artifact-" + suffix).encode(),
        job_id="job-" + suffix,
        purpose="research_lab.provider_preflight.v2",
        artifact_kind="provider_response",
    )


def test_full_vault_reclaims_only_oldest_stale_orphan(monkeypatch):
    monkeypatch.setattr(vault_module, "MAX_IN_MEMORY_ARTIFACTS", 2)
    clock = Clock()
    vault = _vault(clock)
    first = _seal(vault, "first")
    clock.now += timedelta(seconds=601)
    second = _seal(vault, "second")

    third = _seal(vault, "third")

    with pytest.raises(ArtifactVaultV2Error, match="unavailable"):
        vault.descriptor(first["artifact_id"])
    assert vault.descriptor(second["artifact_id"])
    assert vault.descriptor(third["artifact_id"])
    capacity = vault.transient_capacity_state()
    assert capacity["transient_artifact_bytes"] > 0
    assert capacity == {
        "transient_artifact_count": 2,
        "maximum_transient_artifacts": 2,
        "transient_artifact_bytes": capacity["transient_artifact_bytes"],
        "maximum_transient_artifact_bytes": (
            vault_module.MAX_IN_MEMORY_ARTIFACT_BYTES
        ),
        "evicted_orphan_count": 1,
    }


def test_full_vault_never_evicts_recent_inflight_artifact(monkeypatch):
    monkeypatch.setattr(vault_module, "MAX_IN_MEMORY_ARTIFACTS", 1)
    clock = Clock()
    vault = _vault(clock)
    first = _seal(vault, "first")

    with pytest.raises(ArtifactVaultV2Error, match="capacity is full"):
        _seal(vault, "second")

    assert vault.descriptor(first["artifact_id"])
    assert vault.transient_capacity_state()["evicted_orphan_count"] == 0


def test_full_vault_enforces_total_encoded_byte_boundary(monkeypatch):
    clock = Clock()
    vault = _vault(clock)
    first = _seal(vault, "first")
    used = vault.transient_capacity_state()["transient_artifact_bytes"]
    monkeypatch.setattr(vault_module, "MAX_IN_MEMORY_ARTIFACT_BYTES", used + 1)

    with pytest.raises(ArtifactVaultV2Error, match="capacity is full"):
        _seal(vault, "second")

    assert vault.descriptor(first["artifact_id"])
    assert vault.transient_capacity_state()["transient_artifact_count"] == 1
    assert vault.transient_capacity_state()["transient_artifact_bytes"] == used


def test_full_vault_never_evicts_old_active_transaction(monkeypatch):
    monkeypatch.setattr(vault_module, "MAX_IN_MEMORY_ARTIFACTS", 1)
    clock = Clock()
    vault = _vault(clock)

    with vault.transient_artifact_transaction():
        first = _seal(vault, "first")
        clock.now += timedelta(seconds=601)
        with pytest.raises(ArtifactVaultV2Error, match="capacity is full"):
            _seal(vault, "second")

    assert vault.descriptor(first["artifact_id"])
    assert vault.transient_capacity_state()["evicted_orphan_count"] == 0
