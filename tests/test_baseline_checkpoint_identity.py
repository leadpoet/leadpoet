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
