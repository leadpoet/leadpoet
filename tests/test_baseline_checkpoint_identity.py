"""Benchmark checkpoints preserve measured work across gateway releases.

A mid-benchmark model change must rescore every ICP with the new model —
old score rows are never carried across a model identity change (artifact
hash, repo commit, or manifest hash). Scoring-configuration and scoring-contract
changes also invalidate progress. Gateway release changes retain the rows and
their immutable receipt/runtime provenance instead of spending the same ICP
budget again.
"""

import json
import sys
import types
from unittest import mock

import gateway.research_lab.scoring_worker as sw


RUNTIME_SHA = "1" * 40
OTHER_RUNTIME_SHA = "2" * 40
SCORING_CONFIG_HASH = "sha256:" + "3" * 64
OTHER_SCORING_CONFIG_HASH = "sha256:" + "4" * 64
PARENT_RECEIPT_HASH = "sha256:" + "5" * 64
OTHER_PARENT_RECEIPT_HASH = "sha256:" + "6" * 64
MODEL_REPO_SHA = "7" * 40
OTHER_MODEL_REPO_SHA = "8" * 40
MODEL_MANIFEST_HASH = "sha256:" + "9" * 64
OTHER_MODEL_MANIFEST_HASH = "sha256:" + "a" * 64


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
        "schema_version": "2.0",
        "artifact_type": "research_lab_private_baseline_scoring_progress",
        "checkpoint_status": "active",
        "benchmark_date": "2026-07-15",
        "rolling_window_hash": "sha256:w1",
        "private_model_artifact_hash": "sha256:a1",
        "gateway_runtime_commit_sha": RUNTIME_SHA,
        "scoring_configuration_hash": SCORING_CONFIG_HASH,
        "attested_parent_receipt_hashes": [PARENT_RECEIPT_HASH],
        "repo_git_sha": MODEL_REPO_SHA,
        "manifest_hash": MODEL_MANIFEST_HASH,
        "completed_icp_count": 1,
        "per_icp_results": [{"icp_ref": "icp-1", "score": 54.0}],
    }
    base.update(over)
    return base


def _load(doc, **kw):
    kw.setdefault("gateway_runtime_commit_sha", RUNTIME_SHA)
    kw.setdefault("scoring_configuration_hash_value", SCORING_CONFIG_HASH)
    with mock.patch.dict(sys.modules, {"boto3": _s3_stub(doc)}):
        return sw._load_baseline_scoring_progress(
            "bucket", "key",
            benchmark_date="2026-07-15", window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1", **kw)


def test_same_identity_reuses():
    rows = _load(
        _doc(),
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
    )
    assert rows and rows[0]["icp_ref"] == "icp-1"


def test_repo_sha_change_discards():
    assert _load(
        _doc(),
        repo_git_sha=OTHER_MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
    ) == []


def test_manifest_change_discards():
    assert _load(
        _doc(),
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=OTHER_MODEL_MANIFEST_HASH,
    ) == []


def test_artifact_hash_change_discards():
    with mock.patch.dict(sys.modules, {"boto3": _s3_stub(_doc())}):
        rows = sw._load_baseline_scoring_progress(
            "bucket", "key", benchmark_date="2026-07-15",
            window_hash="sha256:w1", private_model_artifact_hash="sha256:DIFFERENT",
            gateway_runtime_commit_sha=RUNTIME_SHA,
            scoring_configuration_hash_value=SCORING_CONFIG_HASH,
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
        )
    assert rows == []


def test_legacy_checkpoint_without_runtime_identity_is_rejected():
    doc = _doc()
    doc["schema_version"] = "1.0"
    doc.pop("gateway_runtime_commit_sha")
    doc.pop("scoring_configuration_hash")
    rows = _load(
        doc,
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
    )
    assert rows == []


def test_gateway_runtime_change_reuses_measured_rows():
    attempts = []
    producers = set()
    assert _load(
        _doc(),
        gateway_runtime_commit_sha=OTHER_RUNTIME_SHA,
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
        attempt_ledger_out=attempts,
        producer_runtime_commits_out=producers,
    ) == [{"icp_ref": "icp-1", "score": 54.0}]
    assert producers == {RUNTIME_SHA}
    assert len(attempts) == 1
    assert attempts[0]["gateway_runtime_commit_sha"] == RUNTIME_SHA


def test_scoring_configuration_change_discards():
    assert _load(
        _doc(),
        scoring_configuration_hash_value=OTHER_SCORING_CONFIG_HASH,
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
    ) == []


def test_invalid_expected_runtime_identity_fails_closed():
    with mock.patch.dict(sys.modules, {"boto3": _s3_stub(_doc())}):
        try:
            sw._load_baseline_scoring_progress(
                "bucket",
                "key",
                benchmark_date="2026-07-15",
                window_hash="sha256:w1",
                private_model_artifact_hash="sha256:a1",
                gateway_runtime_commit_sha="short",
                scoring_configuration_hash_value=SCORING_CONFIG_HASH,
                repo_git_sha=MODEL_REPO_SHA,
                manifest_hash=MODEL_MANIFEST_HASH,
            )
        except ValueError as exc:
            assert "runtime commit" in str(exc)
        else:
            raise AssertionError("invalid runtime identity was accepted")


def test_checkpoint_without_attested_parent_receipts_is_rejected():
    doc = _doc()
    doc.pop("attested_parent_receipt_hashes")
    assert _load(
        doc,
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
    ) == []


def test_checkpoint_without_required_structural_identity_is_rejected():
    for field in (
        "artifact_type",
        "checkpoint_status",
        "repo_git_sha",
        "manifest_hash",
        "completed_icp_count",
    ):
        doc = _doc()
        doc.pop(field)
        assert _load(
            doc,
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
        ) == []


def test_checkpoint_with_inconsistent_completed_count_is_rejected():
    for invalid_count in (0, 2, True, "1"):
        assert _load(
            _doc(completed_icp_count=invalid_count),
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
        ) == []


def test_invalid_expected_model_identity_fails_closed():
    for repo_git_sha, manifest_hash, expected_fragment in (
        ("short", MODEL_MANIFEST_HASH, "repository commit"),
        (MODEL_REPO_SHA, "sha256:short", "manifest hash"),
    ):
        try:
            _load(
                _doc(),
                repo_git_sha=repo_git_sha,
                manifest_hash=manifest_hash,
            )
        except ValueError as exc:
            assert expected_fragment in str(exc)
        else:
            raise AssertionError("invalid model identity was accepted")


def test_checkpoint_with_duplicate_or_invalid_parent_receipts_is_rejected():
    assert _load(
        _doc(
            attested_parent_receipt_hashes=[
                PARENT_RECEIPT_HASH,
                PARENT_RECEIPT_HASH,
            ]
        ),
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
    ) == []
    assert _load(
        _doc(attested_parent_receipt_hashes=["not-a-receipt"]),
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
    ) == []


def test_checkpoint_parent_receipts_are_bounded_by_v2_transport_limit():
    oversized = [f"sha256:{index:064x}" for index in range(129)]

    assert _load(
        _doc(attested_parent_receipt_hashes=oversized),
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
    ) == []


def test_checkpoint_resume_restores_attested_parent_receipt_hashes():
    restored: set[str] = set()
    rows = _load(
        _doc(
            attested_parent_receipt_hashes=[
                OTHER_PARENT_RECEIPT_HASH,
                PARENT_RECEIPT_HASH,
            ]
        ),
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
        parent_receipt_hashes_out=restored,
    )

    assert rows
    assert restored == {PARENT_RECEIPT_HASH, OTHER_PARENT_RECEIPT_HASH}


def test_invalidated_checkpoint_cannot_be_resumed():
    assert _load(
        _doc(
            checkpoint_status="invalidated",
            invalidation_reason="globally_all_zero",
        ),
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
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
            ],
            completed_icp_count=2,
        ),
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
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
            ],
            completed_icp_count=39,
        ),
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
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
            gateway_runtime_commit_sha=RUNTIME_SHA,
            scoring_configuration_hash_value=SCORING_CONFIG_HASH,
            rows=rows,
            reason="globally_all_zero",
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
        )
        loaded = sw._load_baseline_scoring_progress(
            "bucket",
            "key",
            benchmark_date="2026-07-15",
            window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1",
            gateway_runtime_commit_sha=RUNTIME_SHA,
            scoring_configuration_hash_value=SCORING_CONFIG_HASH,
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
        )

    doc = json.loads(stored["body"])
    assert digest == sw.canonical_hash(doc)
    assert doc["checkpoint_status"] == "invalidated"
    assert doc["gateway_runtime_commit_sha"] == RUNTIME_SHA
    assert doc["scoring_configuration_hash"] == SCORING_CONFIG_HASH
    assert doc["rejected_icp_count"] == 2
    assert doc["per_icp_results"] == []
    assert loaded == []


def test_active_checkpoint_round_trip_preserves_runtime_provenance():
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

    with mock.patch.dict(sys.modules, {"boto3": stub}):
        digest = sw._store_baseline_scoring_progress(
            "bucket",
            "key",
            benchmark_date="2026-07-15",
            window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1",
            gateway_runtime_commit_sha=RUNTIME_SHA,
            scoring_configuration_hash_value=SCORING_CONFIG_HASH,
            rows=[{"icp_ref": "icp-1", "score": 54.0, "company_count": 1}],
            attested_parent_receipt_hashes=[PARENT_RECEIPT_HASH],
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
        )
        attempts = []
        producers = set()
        loaded = sw._load_baseline_scoring_progress(
            "bucket",
            "key",
            benchmark_date="2026-07-15",
            window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1",
            gateway_runtime_commit_sha=RUNTIME_SHA,
            scoring_configuration_hash_value=SCORING_CONFIG_HASH,
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
            attempt_ledger_out=attempts,
            producer_runtime_commits_out=producers,
        )
        cross_release_loaded = sw._load_baseline_scoring_progress(
            "bucket",
            "key",
            benchmark_date="2026-07-15",
            window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1",
            gateway_runtime_commit_sha=OTHER_RUNTIME_SHA,
            scoring_configuration_hash_value=SCORING_CONFIG_HASH,
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
        )
        changed_contract_loaded = sw._load_baseline_scoring_progress(
            "bucket",
            "key",
            benchmark_date="2026-07-15",
            window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1",
            gateway_runtime_commit_sha=OTHER_RUNTIME_SHA,
            scoring_configuration_hash_value=SCORING_CONFIG_HASH,
            scoring_contract_hash_value="sha256:" + "f" * 64,
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
        )

    doc = json.loads(stored["body"])
    assert digest == sw.canonical_hash(doc)
    assert doc["schema_version"] == "3.0"
    assert doc["gateway_runtime_commit_sha"] == RUNTIME_SHA
    assert doc["producer_gateway_runtime_commits"] == [RUNTIME_SHA]
    assert doc["scoring_configuration_hash"] == SCORING_CONFIG_HASH
    assert doc["scoring_contract_hash"].startswith("sha256:")
    assert doc["provider_cost_base_scope_hash"].startswith("sha256:")
    assert doc["repo_git_sha"] == MODEL_REPO_SHA
    assert doc["manifest_hash"] == MODEL_MANIFEST_HASH
    assert doc["attested_parent_receipt_hashes"] == [PARENT_RECEIPT_HASH]
    assert doc["attempt_ledger"]["settled_attempt_count"] == 1
    assert loaded == [{"icp_ref": "icp-1", "score": 54.0, "company_count": 1}]
    assert producers == {RUNTIME_SHA}
    assert len(attempts) == 1
    assert cross_release_loaded == loaded
    assert changed_contract_loaded == []


def test_unresolved_attempt_ledger_round_trips_across_release():
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
    attempt_rows = []
    for retry_round in (0, 1):
        row = {
            "icp_ref": "icp-1",
            "icp_hash": "hash-1",
            "score": 0.0,
            "company_count": 0,
            "diagnostics": {
                "sourcing_failed": True,
                "runtime_error": {"category": "provider"},
            },
            "_item_index": 1,
            "_retryable": True,
            "_nonempty": False,
            "_runtime_error": "HTTP 500",
            "_retry_backoff_seconds": 0.0,
            sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD: [],
        }
        attempt_rows.append(
            sw._baseline_attempt_ledger_entry(
                row,
                retry_round=retry_round,
                gateway_runtime_commit_sha=RUNTIME_SHA,
            )
        )

    with mock.patch.dict(sys.modules, {"boto3": stub}):
        sw._store_baseline_scoring_progress(
            "bucket",
            "key",
            benchmark_date="2026-07-15",
            window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1",
            gateway_runtime_commit_sha=RUNTIME_SHA,
            scoring_configuration_hash_value=SCORING_CONFIG_HASH,
            rows=[],
            attested_parent_receipt_hashes=[],
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
            attempt_ledger=attempt_rows,
            provider_cost_base_scope_hash="sha256:" + "b" * 64,
        )
        restored_attempts = []
        scope_values = []
        loaded = sw._load_baseline_scoring_progress(
            "bucket",
            "key",
            benchmark_date="2026-07-15",
            window_hash="sha256:w1",
            private_model_artifact_hash="sha256:a1",
            gateway_runtime_commit_sha=OTHER_RUNTIME_SHA,
            scoring_configuration_hash_value=SCORING_CONFIG_HASH,
            repo_git_sha=MODEL_REPO_SHA,
            manifest_hash=MODEL_MANIFEST_HASH,
            attempt_ledger_out=restored_attempts,
            provider_cost_base_scope_out=scope_values,
            benchmark_items=[{"icp_ref": "icp-1", "icp_hash": "hash-1"}],
        )

    assert loaded == []
    assert [entry["retry_round"] for entry in restored_attempts] == [0, 1]
    assert {
        entry["result_row"]["_runtime_error"] for entry in restored_attempts
    } == {"attempt_failed"}
    assert b"HTTP 500" not in stored["body"]
    assert scope_values == ["sha256:" + "b" * 64]


def test_attempt_ledger_gap_is_rejected():
    base = _doc(schema_version="3.0")
    row = {
        "icp_ref": "icp-1",
        "score": 54.0,
        "_item_index": 1,
        "_retryable": False,
        "_nonempty": True,
        "_runtime_error": "",
        "_retry_backoff_seconds": 0.0,
        sw._BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD: [],
    }
    entry = sw._baseline_attempt_ledger_entry(
        row,
        retry_round=1,
        gateway_runtime_commit_sha=RUNTIME_SHA,
    )
    payload = {
        "schema_version": sw._BASELINE_ATTEMPT_LEDGER_SCHEMA_VERSION,
        "settled_attempt_count": 1,
        "entries": [entry],
    }
    base.update(
        scoring_contract_hash=sw._baseline_scoring_contract_hash(),
        provider_cost_base_scope_hash="sha256:" + "b" * 64,
        producer_gateway_runtime_commits=[RUNTIME_SHA],
        attempt_ledger={**payload, "ledger_hash": sw.canonical_hash(payload)},
    )

    assert _load(
        base,
        repo_git_sha=MODEL_REPO_SHA,
        manifest_hash=MODEL_MANIFEST_HASH,
    ) == []


def test_persisted_receipt_roots_merge_with_live_receipts_without_duplicates():
    class ReceiptSource:
        @staticmethod
        def attested_receipts():
            return [
                {"receipt_hash": PARENT_RECEIPT_HASH, "status": "succeeded"},
            ]

    receipts = sw._attested_receipts_with_persisted_roots(
        ReceiptSource(),
        persisted_receipt_hashes=[
            OTHER_PARENT_RECEIPT_HASH,
            PARENT_RECEIPT_HASH,
        ],
    )

    assert receipts == [
        {"receipt_hash": PARENT_RECEIPT_HASH, "status": "succeeded"},
        {"receipt_hash": OTHER_PARENT_RECEIPT_HASH},
    ]
