"""Age-guarded transient eviction: the vault must reclaim orphaned transients
instead of failing every seal closed once full — while never dropping a still
in-flight artifact (one seconds from persistence).

Root cause this guards: inter-enclave sealed transients strand on any
pre-persist failure of execute_scoring_v2, accumulating until the vault hits
MAX_IN_MEMORY_ARTIFACTS and then fails EVERY execution closed, including the
weight-path allocation build.
"""
from datetime import datetime, timedelta, timezone

import pytest

from gateway.tee import artifact_vault_v2 as vault_mod
from gateway.tee.artifact_vault_v2 import (
    ArtifactVaultV2Error,
    EncryptedArtifactVaultV2,
)

MASTER_KEY = bytes(range(32))
BOOT_HASH = "sha256:" + "a" * 64


class _Clock:
    def __init__(self):
        self.now = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + timedelta(seconds=seconds)


def _vault(clock):
    return EncryptedArtifactVaultV2(
        master_key=MASTER_KEY, boot_identity_hash=BOOT_HASH, clock=clock
    )


def _fill(vault, n, clock, kind="orphan"):
    for i in range(n):
        vault.seal(
            b"x" * (i + 1),  # distinct plaintext -> distinct artifact_id
            job_id="job-%d" % i,
            purpose="p",
            artifact_kind=kind,
        )


def test_stale_transients_are_evicted_to_admit_new_seal(monkeypatch):
    # Shrink the cap so the test is fast; keep the real age guard.
    monkeypatch.setattr(vault_mod, "MAX_IN_MEMORY_ARTIFACTS", 4)
    clock = _Clock()
    vault = _vault(clock)
    _fill(vault, 4, clock)                       # 4 orphaned transients, full
    clock.advance(vault_mod.TRANSIENT_EVICTION_MIN_AGE_SECONDS + 1)  # now stale

    # The 5th seal would have raised "capacity is full"; now it evicts one
    # stale orphan and succeeds.
    desc = vault.seal(b"new", job_id="job-new", purpose="p", artifact_kind="k")
    assert desc["artifact_id"]
    assert vault._evicted_transient_count == 1


def test_in_flight_transients_are_never_evicted(monkeypatch):
    monkeypatch.setattr(vault_mod, "MAX_IN_MEMORY_ARTIFACTS", 4)
    clock = _Clock()
    vault = _vault(clock)
    _fill(vault, 4, clock)                        # 4 RECENT transients, full
    # No time advance: all are in-flight (younger than the guard). A new seal
    # must fail closed rather than evict an about-to-persist artifact.
    with pytest.raises(ArtifactVaultV2Error, match="capacity is full"):
        vault.seal(b"new", job_id="j", purpose="p", artifact_kind="k")
    assert vault._evicted_transient_count == 0


def test_eviction_is_oldest_first_and_only_as_needed(monkeypatch):
    monkeypatch.setattr(vault_mod, "MAX_IN_MEMORY_ARTIFACTS", 4)
    clock = _Clock()
    vault = _vault(clock)
    # Seal 4 at staggered ages; only 2 are past the guard.
    old2 = []
    for i in range(2):
        d = vault.seal(b"old%d" % i, job_id="old-%d" % i, purpose="p", artifact_kind="k")
        old2.append(d["artifact_id"])
    clock.advance(vault_mod.TRANSIENT_EVICTION_MIN_AGE_SECONDS + 10)
    for i in range(2):
        vault.seal(b"young%d" % i, job_id="yng-%d" % i, purpose="p", artifact_kind="k")
    # Full (4). New seal frees exactly ONE (the oldest), leaves the rest.
    vault.seal(b"new", job_id="j", purpose="p", artifact_kind="k")
    assert vault._evicted_transient_count == 1
    # The single freed one is the oldest (old2[0]); old2[1] and youngs remain.
    with pytest.raises(ArtifactVaultV2Error):
        vault.descriptor(old2[0])          # evicted -> unavailable
    assert vault.descriptor(old2[1])       # still present
