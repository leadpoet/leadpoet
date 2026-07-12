import pytest

from research_lab.eval.artifacts import PrivateModelArtifactManifest
from research_lab.eval.benchmark import SealedBenchmarkSet
from research_lab.eval.baseline_summary import build_baseline_score_summary
from gateway.research_lab.public_benchmarks import build_benchmark_visibility_split
from gateway.research_lab import scoring_worker as sw
from gateway.research_lab.scoring_worker import (
    _baseline_serving_model_version_doc,
    _build_baseline_health,
    _daily_noise_budget_doc,
    _with_baseline_evaluation_contexts,
)


def _fixture():
    items = []
    summaries = []
    for index in range(10):
        ref = "icp:1:%s" % index
        digest = "sha256:" + ("%x" % index) * 64
        items.append(
            {
                "icp_ref": ref,
                "icp_hash": digest,
                "set_id": 1,
                "day_index": 0,
                "day_rank": index + 1,
                "icp": {"industry": "industry-%s" % index},
            }
        )
        summaries.append(
            {
                "icp_ref": ref,
                "icp_hash": digest,
                "score": float((index + 1) * 3),
                "company_count": 5,
                "diagnostics": {},
            }
        )
    artifact = PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + "1" * 64,
        git_commit_sha="2" * 40,
        image_digest="repo@sha256:" + "3" * 64,
        config_hash="sha256:" + "4" * 64,
        component_registry_version="v1",
        scoring_adapter_version="v1",
        manifest_uri="s3://private/model.json",
        manifest_hash="sha256:" + "5" * 64,
        signature_ref="kms://signature",
        build_id="build-1",
    )
    return artifact, items, summaries


def test_shared_baseline_summary_is_byte_identical_to_existing_host_assembly(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_PUBLIC_SPLIT_UNBIASED", "false")
    artifact, items, summaries = _fixture()
    aggregate_score = sum(item["score"] for item in summaries) / len(summaries)
    baseline_health = _build_baseline_health(
        per_icp_summaries=summaries,
        retried=2,
        recovered=1,
        max_unresolved_icps=2,
    )
    baseline_health["day_jump_points"] = 0.75
    visibility_split = build_benchmark_visibility_split(
        rolling_window_hash="sha256:" + "6" * 64,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=3,
        public_weak_per_day=2,
        public_total_icps=3,
        public_weak_total=2,
    )
    serving = _baseline_serving_model_version_doc(
        artifact=artifact,
        benchmark_date="2026-07-10",
        benchmark_attempt=3,
        rolling_window_hash="sha256:" + "6" * 64,
        evaluation_epoch=42,
    )
    enriched = _with_baseline_evaluation_contexts(
        summaries,
        benchmark_date="2026-07-10",
        benchmark_attempt=3,
        rolling_window_hash="sha256:" + "6" * 64,
        evaluation_epoch=42,
        serving_model_version_hash=serving["version_stamp_hash"],
    )
    noise = _daily_noise_budget_doc(
        benchmark_date="2026-07-10",
        rolling_window_hash="sha256:" + "6" * 64,
        per_icp_summaries=enriched,
        aggregate_score=aggregate_score,
    )
    expected = {
        "schema_version": "1.0",
        "benchmark_quality": "passed",
        "benchmark_attempt": 3,
        "rolling_window_hash": "sha256:" + "6" * 64,
        "serving_model_version": serving,
        "per_icp_summaries": enriched,
        "visibility_split": visibility_split,
        "daily_noise_budget": noise,
        "aggregate_score": aggregate_score,
        "baseline_health": baseline_health,
        "elapsed_seconds": 123.456,
    }

    result = build_baseline_score_summary(
        artifact_manifest=artifact.to_dict(),
        benchmark_date="2026-07-10",
        benchmark_attempt=3,
        rolling_window_hash="sha256:" + "6" * 64,
        evaluation_epoch=42,
        benchmark_items=items,
        per_icp_summaries=summaries,
        public_icps_per_day=3,
        public_weak_per_day=2,
        public_total_icps=3,
        public_weak_total=2,
        retried=2,
        recovered=1,
        max_unresolved_icps=2,
        day_jump_points=0.75,
        elapsed_seconds=123.456,
    )

    assert result["score_summary_doc"] == expected


@pytest.mark.asyncio
async def test_candidate_receipt_binds_exact_baseline_summary_and_company_receipts(monkeypatch):
    artifact, _items, _summaries = _fixture()
    baseline_hash = "sha256:" + "7" * 64
    baseline_root = {"receipt_hash": "sha256:" + "8" * 64}
    baseline_ancestor = {"receipt_hash": "sha256:" + "9" * 64}
    company_receipt = {"receipt_hash": "sha256:" + "a" * 64}
    captured = {}

    async def _resolve(**kwargs):
        assert kwargs == {
            "artifact_kind": "benchmark_score_summary",
            "artifact_ref": "private_benchmark:" + "b" * 64,
            "artifact_hash": baseline_hash,
        }
        return baseline_root, [baseline_ancestor, baseline_root]

    async def _compare(**kwargs):
        captured.update(kwargs)
        return {"status": "matched"}

    monkeypatch.setattr(sw, "resolve_attested_artifact_lineage", _resolve)
    monkeypatch.setattr(sw, "compare_attested_score_bundle", _compare)

    result = await sw._compare_candidate_score_bundle_in_enclave(
        evaluation_epoch=42,
        artifact=artifact,
        benchmark=SealedBenchmarkSet(
            benchmark_id="benchmark-1",
            icp_set_hash="sha256:" + "c" * 64,
            split_ref="split-1",
            item_refs=("icp:1",),
            scoring_version="v1",
        ),
        patch={"patch_hash": "sha256:" + "d" * 64},
        candidate_artifact=artifact,
        per_icp_results=[],
        run_context={"evidence_bundle_refs": []},
        policy={},
        private_holdout_gate={
            "baseline_benchmark_bundle_id": "private_benchmark:" + "b" * 64,
            "baseline_benchmark_hash": baseline_hash,
        },
        expected_score_bundle={"score_bundle_hash": "sha256:" + "e" * 64},
        parent_receipts=[company_receipt],
    )

    assert result == {"status": "matched"}
    assert captured["direct_parent_receipt_hashes"] == sorted(
        [company_receipt["receipt_hash"], baseline_root["receipt_hash"]]
    )
    assert {item["receipt_hash"] for item in captured["parent_receipts"]} == {
        company_receipt["receipt_hash"],
        baseline_root["receipt_hash"],
        baseline_ancestor["receipt_hash"],
    }
    assert captured["evidence_roots"]["baseline_score_summary"] == baseline_hash
