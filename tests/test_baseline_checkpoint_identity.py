"""Benchmark checkpoints are bound to the exact model that wrote them.

A mid-benchmark model change must rescore every ICP with the new model —
old score rows are never carried across a model identity change (artifact
hash, repo commit, or manifest hash). Cost recovery on the rescore comes
from the provider-call cache, not from reused results.
"""

import json
import sys
import types
from unittest import mock

import gateway.research_lab.scoring_worker as sw


def _s3_stub(doc):
    body = mock.Mock()
    body.read.return_value = json.dumps(doc).encode()
    s3 = mock.Mock()
    s3.get_object.return_value = {"Body": body}
    stub = types.ModuleType("boto3")
    stub.client = lambda *a, **k: s3
    return stub


def _doc(**over):
    base = {
        "artifact_type": "research_lab_private_baseline_scoring_progress",
        "benchmark_date": "2026-07-15",
        "rolling_window_hash": "sha256:w1",
        "private_model_artifact_hash": "sha256:a1",
        "repo_git_sha": "abc123",
        "manifest_hash": "sha256:m1",
        "per_icp_results": [{"icp_ref": "icp-1", "score": 54.0}],
    }
    base.update(over)
    return base


def _load(doc, **kw):
    with mock.patch.dict(sys.modules, {"boto3": _s3_stub(doc)}):
        return sw._load_baseline_scoring_progress(
            "bucket", "key",
            benchmark_date="2026-07-15", window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1", **kw)


def test_same_identity_reuses():
    rows = _load(_doc(), repo_git_sha="abc123", manifest_hash="sha256:m1")
    assert rows and rows[0]["icp_ref"] == "icp-1"


def test_repo_sha_change_discards():
    assert _load(_doc(), repo_git_sha="def456", manifest_hash="sha256:m1") == []


def test_manifest_change_discards():
    assert _load(_doc(), repo_git_sha="abc123", manifest_hash="sha256:m2") == []


def test_artifact_hash_change_discards():
    with mock.patch.dict(sys.modules, {"boto3": _s3_stub(_doc())}):
        rows = sw._load_baseline_scoring_progress(
            "bucket", "key", benchmark_date="2026-07-15",
            window_hash="sha256:w1", private_model_artifact_hash="sha256:DIFFERENT")
    assert rows == []


def test_legacy_checkpoint_without_new_fields_still_loads():
    doc = _doc(); doc.pop("repo_git_sha"); doc.pop("manifest_hash")
    rows = _load(doc, repo_git_sha="abc123", manifest_hash="sha256:m1")
    assert rows  # backward compatible: old docs lack the fields, not rejected


def test_invalidated_checkpoint_cannot_be_resumed():
    assert _load(
        _doc(
            checkpoint_status="invalidated",
            invalidation_reason="globally_all_zero",
        ),
        repo_git_sha="abc123",
        manifest_hash="sha256:m1",
    ) == []


def test_sourcing_failed_checkpoint_rows_are_not_reused():
    rows = _load(
        _doc(
            per_icp_results=[
                {
                    "icp_ref": "icp-failed",
                    "score": 0.0,
                    "company_count": 0,
                    "diagnostics": {"sourcing_failed": True},
                },
                {
                    "icp_ref": "icp-valid",
                    "score": 0.0,
                    "company_count": 1,
                    "diagnostics": {"sourcing_failed": False},
                },
            ]
        ),
        repo_git_sha="abc123",
        manifest_hash="sha256:m1",
    )

    assert [row["icp_ref"] for row in rows] == ["icp-valid"]


def test_partial_39_of_40_empty_checkpoint_cannot_poison_final_aggregate():
    rows = _load(
        _doc(
            per_icp_results=[
                {
                    "icp_ref": f"icp-{index}",
                    "score": 0.0,
                    "company_count": 0,
                    "diagnostics": {"sourcing_failed": True},
                }
                for index in range(39)
            ]
        ),
        repo_git_sha="abc123",
        manifest_hash="sha256:m1",
    )

    assert rows == []


def test_complete_distribution_requires_every_unique_icp():
    benchmark_items = [{"icp_ref": "icp-1"}, {"icp_ref": "icp-2"}]
    assert sw._baseline_distribution_complete(
        [{"icp_ref": "icp-1"}, {"icp_ref": "icp-2"}],
        benchmark_items,
    )
    assert not sw._baseline_distribution_complete(
        [{"icp_ref": "icp-1"}],
        benchmark_items,
    )
    assert not sw._baseline_distribution_complete(
        [{"icp_ref": "icp-1"}, {"icp_ref": "icp-1"}],
        benchmark_items,
    )


def test_globally_all_zero_invalidation_is_durable_and_non_reusable():
    stored = {}
    s3 = mock.Mock()

    def put_object(**kwargs):
        stored["body"] = bytes(kwargs["Body"])

    def get_object(**_kwargs):
        body = mock.Mock()
        body.read.return_value = stored["body"]
        return {"Body": body}

    s3.put_object.side_effect = put_object
    s3.get_object.side_effect = get_object
    stub = types.ModuleType("boto3")
    stub.client = lambda *args, **kwargs: s3
    rows = [
        {"icp_ref": "icp-1", "score": 0.0, "company_count": 0},
        {"icp_ref": "icp-2", "score": 0.0, "company_count": 0},
    ]

    with mock.patch.dict(sys.modules, {"boto3": stub}):
        digest = sw._invalidate_baseline_scoring_progress(
            "bucket",
            "key",
            benchmark_date="2026-07-15",
            window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1",
            rows=rows,
            reason="globally_all_zero",
            repo_git_sha="abc123",
            manifest_hash="sha256:m1",
        )
        loaded = sw._load_baseline_scoring_progress(
            "bucket",
            "key",
            benchmark_date="2026-07-15",
            window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1",
            repo_git_sha="abc123",
            manifest_hash="sha256:m1",
        )

    doc = json.loads(stored["body"])
    assert digest == sw.canonical_hash(doc)
    assert doc["checkpoint_status"] == "invalidated"
    assert doc["rejected_icp_count"] == 2
    assert doc["per_icp_results"] == []
    assert loaded == []
