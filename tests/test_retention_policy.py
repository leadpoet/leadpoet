"""Tests for the P20 retention/erasure lifecycle (trajectoryimprovements.md)."""

from __future__ import annotations

from typing import Any

import pytest

from research_lab import retention_policy as rp


def test_policy_is_versioned_and_covers_every_trace_channel():
    assert rp.RETENTION_POLICY_VERSION  # dated, versioned in-repo
    assert set(rp.TRACE_PREFIX_RETENTION_POLICY) == {
        "trajectories",
        "telemetry",
        "scorer-traces",
        "incontainer-traces",
        "diagnostics",
    }
    # The corpus raw layer is never auto-expired — deletion only via erasure.
    for channel, policy in rp.TRACE_PREFIX_RETENTION_POLICY.items():
        assert policy["expire_days"] is None, channel


def test_rendered_rules_are_deterministic_and_scoped():
    rules = rp.render_s3_lifecycle_rules("research-lab")
    assert rules == rp.render_s3_lifecycle_rules("research-lab")
    by_prefix = {rule["Filter"]["Prefix"]: rule for rule in rules}
    assert "research-lab/trajectories/" in by_prefix
    trajectory_rule = by_prefix["research-lab/trajectories/"]
    assert "Expiration" not in trajectory_rule  # retain indefinitely
    assert trajectory_rule["Transitions"][0]["StorageClass"] == "STANDARD_IA"
    assert rp.RETENTION_POLICY_VERSION in trajectory_rule["ID"]


def test_audit_flags_missing_and_out_of_band_rules():
    expected = rp.render_s3_lifecycle_rules("research-lab")
    # Fully compliant bucket.
    report = rp.audit_bucket_lifecycle(expected, key_prefix="research-lab")
    assert report["compliant"] is True

    # A cost-driven cleanup rule someone added in the console: expires the
    # raw trajectory layer. Must be flagged as the most destructive class.
    rogue = {
        "ID": "cleanup-old-stuff",
        "Filter": {"Prefix": "research-lab/trajectories/"},
        "Status": "Enabled",
        "Expiration": {"Days": 30},
    }
    report = rp.audit_bucket_lifecycle([*expected, rogue], key_prefix="research-lab")
    assert report["compliant"] is False
    assert report["out_of_band_rules"] == [
        {"rule_id": "cleanup-old-stuff", "prefix": "research-lab/trajectories/"}
    ]
    assert report["unexpected_expirations"] == report["out_of_band_rules"]

    # Missing rules are findings too.
    report = rp.audit_bucket_lifecycle([], key_prefix="research-lab")
    assert report["compliant"] is False
    assert len(report["missing_rules"]) == len(expected)

    # A rule on an unrelated prefix is not our business.
    unrelated = {
        "ID": "tmp-cleanup",
        "Filter": {"Prefix": "tmp/"},
        "Status": "Enabled",
        "Expiration": {"Days": 1},
    }
    report = rp.audit_bucket_lifecycle([*expected, unrelated], key_prefix="research-lab")
    assert report["compliant"] is True


def test_verification_state_for_bundle_attests_hash_after_erasure():
    erased = {
        "verification_state": "content_deleted",
        "bundle_hash": "sha256:bb",
        "deletion_request_ref": "deletion_request:dr-1",
    }
    assert rp.verification_state_for_bundle(erased) == "hash_attested"
    # content_deleted WITHOUT a valid request/hash never attests.
    assert (
        rp.verification_state_for_bundle(
            {"verification_state": "content_deleted", "bundle_hash": ""}
        )
        == "content_deleted"
    )
    assert rp.verification_state_for_bundle({"verification_state": "active"}) == "active"


# ---------------------------------------------------------------------------
# erasure job
# ---------------------------------------------------------------------------


class _EraseStore:
    def __init__(self, bundle):
        self.bundle = bundle

    async def select_one(self, table, *, columns="*", filters=()):
        return dict(self.bundle)


class _DeletingS3:
    def __init__(self):
        self.deleted: list[str] = []

    def delete_object(self, Bucket, Key):
        self.deleted.append(f"s3://{Bucket}/{Key}")


def _bundle():
    return {
        "bundle_id": "eb-1",
        "bundle_hash": "sha256:bundle",
        "verification_state": "active",
        "snapshots": [
            {
                "snapshot_kind": "per_icp_score_evidence",
                "incontainer_trace_ref": "s3://bucket/incontainer/icp-1.json",
                "incontainer_trace_sha256": "sha256:ii",
                "scorer_trace_ref": "s3://bucket/scorer-traces/icp-1.json",
                "scorer_trace_sha256": "sha256:ss",
            }
        ],
        "bundle_doc": {},
    }


async def test_erase_bundle_dry_run_targets_but_deletes_nothing():
    from gateway.research_lab.trace_reconciler import erase_evidence_bundle_content

    result = await erase_evidence_bundle_content(
        "eb-1",
        deletion_request_ref="deletion_request:dr-1",
        store=_EraseStore(_bundle()),
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert set(result["objects_targeted"]) == {
        "s3://bucket/incontainer/icp-1.json",
        "s3://bucket/scorer-traces/icp-1.json",
    }
    assert result["objects_deleted"] == []
    assert result["bundle_hash_retained"] == "sha256:bundle"


async def test_erase_bundle_deletes_and_flips_state():
    from gateway.research_lab.trace_reconciler import erase_evidence_bundle_content

    s3 = _DeletingS3()
    updates: list[tuple[str, dict, tuple]] = []

    async def fake_update_row(table, values, *, filters):
        updates.append((table, values, tuple(filters)))
        return {"bundle_id": "eb-1", **values}

    result = await erase_evidence_bundle_content(
        "eb-1",
        deletion_request_ref="deletion_request:dr-1",
        store=_EraseStore(_bundle()),
        s3_client=s3,
        update_row=fake_update_row,
        dry_run=False,
    )
    assert result["status"] == "content_deleted"
    assert set(s3.deleted) == set(result["objects_targeted"])
    table, values, filters = updates[0]
    assert table == "evidence_bundles"
    assert values == {
        "verification_state": "content_deleted",
        "deletion_request_ref": "deletion_request:dr-1",
    }
    assert filters == (("bundle_id", "eb-1"),)


async def test_erase_bundle_requires_deletion_request():
    from gateway.research_lab.trace_reconciler import erase_evidence_bundle_content

    with pytest.raises(ValueError):
        await erase_evidence_bundle_content(
            "eb-1", deletion_request_ref="", store=_EraseStore(_bundle())
        )
