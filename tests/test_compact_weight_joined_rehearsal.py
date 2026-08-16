"""Joined regression for the activated compact weight publication path."""

from __future__ import annotations

import asyncio
from pathlib import Path
import re
import subprocess
import sys

import pytest

from leadpoet_canonical.auditor_latest_verified_bundle_v2 import (
    AuditorLatestVerifiedBundleV2Error,
    verified_bundle_projection_v2,
)


REHEARSAL_ROOT = Path(__file__).resolve().parent / "restart_rehearsal"
if str(REHEARSAL_ROOT) not in sys.path:
    sys.path.insert(0, str(REHEARSAL_ROOT))

import compact_weight_joined_runner as compact_runner
from compact_weight_joined_runner import exercise_compact_weight_joined_path


def _verified_bundle(weights_hash: str) -> dict:
    return {
        "epoch_id": 30_000,
        "netuid": 71,
        "block": 10_800_340,
        "uids": [1, 2],
        "weights_u16": [65_535, 32_768],
        "weights_hash": weights_hash,
        "bundle_hash": "sha256:" + "b" * 64,
        "authority_stage": "finalized",
        "validator_hotkey": "5" * 48,
        "receipt_graph_hash": "sha256:" + "c" * 64,
    }


def test_latest_verified_projection_normalizes_compact_raw_weights_digest():
    projection = verified_bundle_projection_v2(_verified_bundle("A" * 64))

    assert projection["weights_hash"] == "sha256:" + "a" * 64


@pytest.mark.parametrize("weights_hash", ["a" * 63, "g" * 64, "sha256:" + "a" * 63])
def test_latest_verified_projection_rejects_malformed_compact_weights_digest(
    weights_hash: str,
):
    with pytest.raises(AuditorLatestVerifiedBundleV2Error, match="weights hash"):
        verified_bundle_projection_v2(_verified_bundle(weights_hash))


def test_compact_weight_joined_path_uses_production_lifecycle_and_recovers(
    monkeypatch,
):
    monkeypatch.setenv("REHEARSAL_SOURCE_ROOT", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", "f" * 40)
    # Earlier joined scenarios use asyncio.run(), which clears the process's
    # current loop on Python 3.9 before the validator module is imported.
    asyncio.run(asyncio.sleep(0))
    evidence = exercise_compact_weight_joined_path()

    for field in (
        "production_allocation_guard",
        "production_primary_compact_lifecycle",
        "gateway_compact_submit_persist_get_finalize",
        "real_epoch_evidence_endpoint",
        "stateful_epoch_evidence_persisted",
        "stateful_epoch_evidence_readback_exact",
        "cutover_authority_db_boundary_exact",
        "release_lineage_file_archive_boundary_exact",
        "compact_ancestry_checkpoint_persistence",
        "primary_auditor_byte_identity",
        "auditor_verified_cache_replay",
        "auditor_submission_success",
        "auditor_last_update_advanced",
        "auditor_finalized_vector_readback_equal",
        "same_epoch_compact_journal_recovered",
        "same_epoch_compact_fresh_scan_recovered",
        "compact_finalization_job_ids_scan_derived",
        "compact_mismatched_recovery_conflict",
        "next_epoch_compact_journal_retired",
    ):
        assert evidence[field] is True
    assert evidence["compact_fresh_scan_recovery_writes"] == 0
    assert evidence["independent_auditor_count"] == 2
    assert evidence["independent_auditor_submission_count"] == 2
    assert len(evidence["auditor_submission_states"]) == 2
    assert {
        item["vector_hash"] for item in evidence["auditor_submission_states"]
    } == {evidence["primary_auditor_vector_hash"]}
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}", evidence["primary_auditor_vector_hash"]
    )
    assert all(
        re.fullmatch(r"sha256:[0-9a-f]{64}", item["vector_hash"])
        for item in evidence["auditor_submission_states"]
    )
    assert all(
        item["last_update"] > 10_800_000
        for item in evidence["auditor_submission_states"]
    )
    assert evidence["real_chain_broadcast_adapted"] is True
    assert evidence["physical_chain_last_update_vector_readback_unadaptable"] is True


def test_compact_runner_uses_explicit_flat_harness_candidate_identity(
    monkeypatch,
):
    source_root = Path(__file__).resolve().parents[1]
    candidate_sha = "e" * 40
    monkeypatch.setenv("REHEARSAL_SOURCE_ROOT", str(source_root))
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", candidate_sha)

    def reject_git(*_args, **_kwargs):
        raise AssertionError("explicit candidate identity must not invoke git")

    monkeypatch.setattr(subprocess, "run", reject_git)

    assert compact_runner._source_root() == source_root.resolve()
    assert compact_runner._candidate_sha() == candidate_sha
