"""Tests for the worker-side docker replay dev evaluator (§6.3 activation).

Covers:
- factory gating (flag off / URI missing / configured),
- local snapshot-set resolution and the missing-manifest guard,
- dev ICP payload loading bound to the manifest icp_set_hash (tamper guard),
- end-to-end candidate evaluation through the runner seam (fake docker
  invocation, real snapshot store + evaluate_dev arithmetic) and its
  determinism,
- immutable-digest requirement and best-effort failure shape.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.research_lab.dev_eval_runner import (
    AttestedReplayDevEvaluatorV2,
    DEV_EVAL_ENABLED_ENV,
    DevEvalRunnerError,
    DockerReplayDevEvaluator,
    build_code_edit_dev_evaluator,
    ensure_local_snapshot_set,
    load_verified_dev_items,
)
from gateway.tee.source_bundle_v2 import build_source_bundle_v2
from research_lab.canonical import sha256_json
from research_lab.eval import PrivateModelArtifactManifest, build_local_private_artifact_manifest
from research_lab.eval.dev_eval import DEV_SCORE_VERSION, compute_dev_set_hash
from research_lab.eval.snapshot_store import (
    MODE_RECORD,
    MODE_REPLAY,
    POINTER_NAME,
    SNAPSHOT_URI_ENV,
    ProviderSnapshotStore,
    build_snapshot_pointer_document,
    build_snapshot_request,
)
from research_lab.eval.provider_evidence_cache import (
    EVIDENCE_CACHE_SCHEMA_VERSION,
    icp_evidence_cache_key,
)

IMAGE_DIGEST = "123456789.dkr.ecr.test/model@sha256:" + "a" * 64

SCRAPINGDOG_URL = (
    "https://api.scrapingdog.com/linkedin?type=company&linkId={link_id}&api_key={key}"
)


@pytest.fixture(autouse=True)
def _clear_dev_env(monkeypatch):
    monkeypatch.delenv(DEV_EVAL_ENABLED_ENV, raising=False)
    monkeypatch.delenv(SNAPSHOT_URI_ENV, raising=False)


def _dev_icp(index: int) -> dict:
    return {
        "icp_id": f"dev-{index}",
        "industry": "Software Development",
        "sub_industry": "DevOps Tooling",
        "product_service": "CI/CD platform",
        "geography": "United States",
        "country": "United States",
        "employee_count": "51-200",
        "intent_signals": [f"Hiring a DevOps engineer wave {index}"],
        "intent_signal": f"Hiring a DevOps engineer wave {index}",
    }


def _dev_items(count: int) -> list[dict]:
    items = []
    for index in range(count):
        icp = _dev_icp(index)
        items.append(
            {
                "icp": icp,
                "icp_ref": f"dev_set:{index}",
                "icp_hash": sha256_json({"icp": icp}),
            }
        )
    return items


def _rich_company(index: int = 0) -> dict:
    return {
        "company_name": f"Acme {index}",
        "company_website": f"https://acme-{index}.test",
        "industry": "Software Development",
        "sub_industry": "DevOps Tooling",
        "employee_count": "51-200",
        "country": "United States",
        "description": "CI/CD platform for DevOps teams",
        "intent_signals": [
            {
                "source": "job_board",
                "description": "Hiring a DevOps engineer to build pipelines",
                "url": f"https://acme-{index}.test/jobs/1",
                "date": "2026-05-01",
            }
        ],
    }


def _write_snapshot_set(tmp_path, items: list[dict]) -> str:
    """Record a complete snapshot set (snapshots + manifest + dev ICPs)."""
    assert len(items) == 8
    root = str(tmp_path / "snapshot_set")
    store = ProviderSnapshotStore(root, mode=MODE_RECORD)
    for item in items:
        body = json.dumps({"companies": [_rich_company(0), _rich_company(1)]})
        request = build_snapshot_request(
            "GET", SCRAPINGDOG_URL.format(link_id=item["icp"]["icp_id"], key="RECORDKEY")
        )
        store.record_response(request, status=200, body_text=body)
    store.write_dev_icp_items(items)
    manifest = store.build_manifest(
        icp_set_hash=compute_dev_set_hash(items),
        dev_set_manifest={"manifest_type": "research_lab_dev_icp_set"},
        recorded_at="2026-07-06T00:00:00Z",
        provenance={
            "champion_image_digest": IMAGE_DIGEST,
            "source_commit": "a" * 40,
            "model_config_hash": "sha256:" + "b" * 64,
            "provider_model_ids": ["test/provider-model"],
            "replay_output_hashes": [
                {"icp_hash": item["icp_hash"], "output_hash": "sha256:" + "c" * 64}
                for item in items
            ],
        },
    )
    store.write_manifest(manifest)
    store.write_ready_document(store.build_ready_document(manifest))
    return root


def _candidate(node_id: str = "node-1", image_digest: str = IMAGE_DIGEST):
    return SimpleNamespace(
        node_id=node_id,
        build=SimpleNamespace(
            candidate_model_manifest=SimpleNamespace(image_digest=image_digest)
        ),
    )


def _fake_docker_runner(companies_by_icp_id: dict[str, list[dict]]):
    calls: list[dict] = []

    def _run(*, image_digest, icp, context, snapshot_dir, timeout_seconds):
        calls.append(
            {
                "image_digest": image_digest,
                "icp_id": icp.get("icp_id"),
                "snapshot_dir": str(snapshot_dir),
                "timeout_seconds": timeout_seconds,
                "context_dev_eval": context.get("dev_eval"),
            }
        )
        return companies_by_icp_id.get(str(icp.get("icp_id")), [])

    _run.calls = calls
    return _run


# ---------------------------------------------------------------------------
# Factory gating
# ---------------------------------------------------------------------------


def test_factory_returns_none_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.setenv(SNAPSHOT_URI_ENV, str(tmp_path))
    assert build_code_edit_dev_evaluator() is None


def test_factory_returns_none_without_snapshot_uri(monkeypatch):
    monkeypatch.setenv(DEV_EVAL_ENABLED_ENV, "true")
    assert build_code_edit_dev_evaluator() is None


def test_factory_returns_evaluator_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv(DEV_EVAL_ENABLED_ENV, "true")
    monkeypatch.setenv(SNAPSHOT_URI_ENV, str(tmp_path / "snapshot_set"))
    evaluator = build_code_edit_dev_evaluator()
    assert isinstance(evaluator, DockerReplayDevEvaluator)


# ---------------------------------------------------------------------------
# Snapshot-set resolution + dev ICP payload guard
# ---------------------------------------------------------------------------


def test_ensure_local_snapshot_set_uses_local_dir_in_place(tmp_path):
    items = _dev_items(8)
    root = _write_snapshot_set(tmp_path, items)
    assert ensure_local_snapshot_set(root) == __import__("pathlib").Path(root)


def test_ensure_local_snapshot_set_requires_manifest(tmp_path):
    empty = tmp_path / "empty_set"
    empty.mkdir()
    with pytest.raises(DevEvalRunnerError, match="not READY|ready_missing"):
        ensure_local_snapshot_set(str(empty))


def _write_local_pointer(base, target: str, *, manifest_hash: str = "", ready_hash: str = ""):
    target_store = ProviderSnapshotStore(target, mode=MODE_REPLAY)
    manifest = target_store.load_manifest() or {}
    ready = target_store.load_ready_document() or {}
    pointer = build_snapshot_pointer_document(
        snapshot_uri=target,
        manifest_hash=manifest_hash or str(manifest.get("manifest_hash") or ""),
        ready_hash=ready_hash or str(ready.get("ready_hash") or ""),
        recorded_at=str(manifest.get("recorded_at") or ""),
    )
    path = base / POINTER_NAME
    path.write_text(json.dumps(pointer), encoding="utf-8")
    return path


def test_local_current_pointer_resolves_verified_immutable_snapshot(tmp_path):
    base = tmp_path / "published"
    base.mkdir()
    target = _write_snapshot_set(base, _dev_items(8))
    pointer = _write_local_pointer(base, target)
    assert ensure_local_snapshot_set(str(pointer)) == __import__("pathlib").Path(target)


def test_snapshot_pointer_rejects_tampering_outside_target_and_mixed_vintages(tmp_path):
    base = tmp_path / "published"
    base.mkdir()
    target = _write_snapshot_set(base, _dev_items(8))

    pointer = _write_local_pointer(base, target)
    tampered = json.loads(pointer.read_text(encoding="utf-8"))
    tampered["recorded_at"] = "tampered"
    pointer.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DevEvalRunnerError, match="pointer_hash_mismatch"):
        ensure_local_snapshot_set(str(pointer))

    outside = tmp_path / "outside"
    outside_target = _write_snapshot_set(outside, _dev_items(8))
    pointer = _write_local_pointer(base, outside_target)
    with pytest.raises(DevEvalRunnerError, match="pointer_target_outside_base"):
        ensure_local_snapshot_set(str(pointer))

    pointer = _write_local_pointer(
        base,
        target,
        manifest_hash="sha256:" + "f" * 64,
    )
    with pytest.raises(DevEvalRunnerError, match="manifest hash does not match"):
        ensure_local_snapshot_set(str(pointer))


def test_load_verified_dev_items_round_trip_and_tamper_guard(tmp_path):
    items = _dev_items(8)
    root = _write_snapshot_set(tmp_path, items)
    replay = ProviderSnapshotStore(root, mode=MODE_REPLAY)
    assert load_verified_dev_items(replay) == [dict(item) for item in items]

    # Swap in a different ICP set without re-recording the manifest.
    tampered = ProviderSnapshotStore(root, mode=MODE_RECORD)
    tampered.write_dev_icp_items(_dev_items(7))
    with pytest.raises(DevEvalRunnerError, match="icp_set_hash"):
        load_verified_dev_items(replay)


def test_load_verified_dev_items_requires_payloads(tmp_path):
    items = _dev_items(8)
    root = _write_snapshot_set(tmp_path, items)
    (__import__("pathlib").Path(root) / "dev_icps.json").unlink()
    replay = ProviderSnapshotStore(root, mode=MODE_REPLAY)
    with pytest.raises(DevEvalRunnerError, match="dev ICP payloads"):
        load_verified_dev_items(replay)


# ---------------------------------------------------------------------------
# End-to-end evaluation through the runner seam
# ---------------------------------------------------------------------------


async def test_evaluator_scores_candidate_and_is_deterministic(tmp_path):
    items = _dev_items(8)
    root = _write_snapshot_set(tmp_path, items)
    companies = {
        "dev-0": [_rich_company(0), _rich_company(1)],
        "dev-1": [_rich_company(2)],
        "dev-2": [],
    }
    fake_docker = _fake_docker_runner(companies)
    evaluator = DockerReplayDevEvaluator(
        snapshot_uri=root, run_icp_in_docker=fake_docker
    )

    first = await evaluator(_candidate())
    second = await evaluator(_candidate())

    assert first == second
    assert first["dev_score_version"] == DEV_SCORE_VERSION
    assert first["ranking_only"] is True
    assert first["icp_count"] == 8
    assert len(first["per_icp"]) == 8
    assert first["aggregate_dev_score"] > 0.0
    # Empty company sets are legitimate zero scores, not infrastructure failures.
    assert first["failure_count"] == 0
    assert first["zero_output_count"] == 6
    assert first["eligible"] is True
    # The fake docker seam received the candidate image and the local dir.
    assert all(call["image_digest"] == IMAGE_DIGEST for call in fake_docker.calls)
    assert all(call["snapshot_dir"] == root for call in fake_docker.calls)
    assert all(call["context_dev_eval"] is True for call in fake_docker.calls)


@pytest.mark.asyncio
async def test_attested_evaluator_preserves_legacy_result_and_requests_candidate_test(
    monkeypatch, tmp_path
):
    items = _dev_items(8)
    root = _write_snapshot_set(tmp_path, items)
    source = tmp_path / "candidate-source"
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def run_icp(icp, context):\n    return []\n",
        encoding="utf-8",
    )
    artifact = PrivateModelArtifactManifest.from_mapping(
        build_local_private_artifact_manifest(
            source_path=source,
            git_commit_sha="a" * 40,
            image_digest=IMAGE_DIGEST,
            manifest_uri="s3://private/candidate.json",
            signature_ref="kms:signature",
            component_registry_version="1",
            scoring_adapter_version="1",
        )
    )
    candidate = SimpleNamespace(
        node_id="node-attested",
        iteration=3,
        draft=SimpleNamespace(lane="conservative"),
        build=SimpleNamespace(candidate_model_manifest=artifact),
    )
    fake_runner = _fake_docker_runner(
        {
            item["icp"]["icp_id"]: [_rich_company(index)]
            for index, item in enumerate(items)
        }
    )
    legacy = DockerReplayDevEvaluator(
        snapshot_uri=root,
        run_icp_in_docker=fake_runner,
    )
    expected = await legacy(candidate)
    source_bundle = build_source_bundle_v2(source)

    async def measured_source_bundle(_artifact, *, timeout_seconds):
        assert timeout_seconds >= 30
        return source_bundle

    monkeypatch.setattr(
        "gateway.research_lab.model_authority_v2.source_bundle_for_artifact_v2",
        measured_source_bundle,
    )
    observed = {}
    measured_result = {}

    async def execute(**kwargs):
        observed.update(kwargs)
        overlay_hash = sha256_json({})
        cohort_hash = kwargs["payload"]["cohort_hash"]
        measured_result.update(
            {
                **expected,
                "evaluation_mode": "replay",
                "overlay_hash": overlay_hash,
                "cohort_hash": cohort_hash,
            }
        )
        measured_result["score_commitment"] = sha256_json(
            {
                "schema_version": "research_lab.git_tree_dev_score_commitment.v1",
                "dev_score_version": expected["dev_score_version"],
                "dev_set_hash": expected["dev_set_hash"],
                "snapshot_manifest_hash": expected[
                    "snapshot_manifest_hash"
                ],
                "miss_policy": expected["miss_policy"],
                "evaluation_mode": "replay",
                "overlay_hash": overlay_hash,
                "cohort_hash": cohort_hash,
            }
        )
        root_hash = "sha256:" + "9" * 64
        return {
            "result": dict(measured_result),
            "receipt_graph": {
                "root_receipt_hash": root_hash,
                "receipts": [
                    {
                        "receipt_hash": root_hash,
                        "output_root": sha256_json(measured_result),
                    }
                ],
            },
        }

    evaluator = AttestedReplayDevEvaluatorV2(
        epoch_id=24000,
        worker_index=4,
        snapshot_uri=root,
        execute=execute,
    )
    result = await evaluator(candidate)

    assert result["result"] == measured_result
    assert result["result"]["cohort_hash"] == observed["payload"][
        "cohort_hash"
    ]
    assert result["result"]["evaluation_mode"] == "replay"
    assert result["result"]["overlay_hash"] == sha256_json({})
    assert result["result"]["aggregate_dev_score"] == expected[
        "aggregate_dev_score"
    ]
    assert observed["operation"] == "run_dev_replay_v2"
    assert observed["purpose"] == "research_lab.candidate_test.v2"
    assert observed["epoch_id"] == 24000
    assert observed["worker_index"] == 4
    assert observed["payload"]["run_label"] == "node-attested"
    assert observed["payload"]["snapshot_manifest_hash"] == expected[
        "snapshot_manifest_hash"
    ]
    assert observed["payload"]["source_bundle"] == source_bundle


def test_attested_evaluator_rejects_tampered_cohort_commitment():
    result = {
        "dev_set_hash": "sha256:" + "1" * 64,
        "snapshot_manifest_hash": "sha256:" + "2" * 64,
        "evaluation_mode": "replay",
        "overlay_hash": sha256_json({}),
        "cohort_hash": "sha256:" + "3" * 64,
    }
    root_hash = "sha256:" + "4" * 64
    outcome = {
        "result": result,
        "receipt_graph": {
            "root_receipt_hash": root_hash,
            "receipts": [
                {
                    "receipt_hash": root_hash,
                    "output_root": sha256_json(result),
                }
            ],
        },
    }

    with pytest.raises(DevEvalRunnerError, match="cohort commitment differs"):
        AttestedReplayDevEvaluatorV2._validate_outcome(
            outcome=outcome,
            manifest={
                "icp_set_hash": result["dev_set_hash"],
                "manifest_hash": result["snapshot_manifest_hash"],
            },
            evaluation_mode="replay",
            overlay_hash=sha256_json({}),
            cohort_hash="sha256:" + "5" * 64,
        )


async def test_evaluator_rejects_mutable_image_reference(tmp_path):
    items = _dev_items(8)
    root = _write_snapshot_set(tmp_path, items)
    evaluator = DockerReplayDevEvaluator(
        snapshot_uri=root, run_icp_in_docker=_fake_docker_runner({})
    )
    with pytest.raises(DevEvalRunnerError, match="immutable"):
        await evaluator(_candidate(image_digest="registry/model:latest"))


async def test_snapshot_preparation_is_lazy_and_single_flight(
    tmp_path, monkeypatch
):
    from gateway.research_lab import dev_eval_runner as runner_module

    calls = {"prepare": 0, "load": 0}
    items = _dev_items(8)

    def prepare_snapshot(uri, *, cache_root=None):
        assert uri == "s3://private/snapshot"
        assert cache_root == tmp_path / "cache"
        calls["prepare"] += 1
        return tmp_path

    def load_items(store):
        assert isinstance(store, ProviderSnapshotStore)
        calls["load"] += 1
        return list(items)

    monkeypatch.setattr(runner_module, "ensure_local_snapshot_set", prepare_snapshot)
    monkeypatch.setattr(runner_module, "load_verified_dev_items", load_items)
    evaluator = DockerReplayDevEvaluator(
        snapshot_uri="s3://private/snapshot",
        cache_root=tmp_path / "cache",
    )

    assert evaluator._prepare_lock is None
    prepared = await asyncio.gather(
        evaluator._ensure_prepared(),
        evaluator._ensure_prepared(),
        evaluator._ensure_prepared(),
    )

    assert calls == {"prepare": 1, "load": 1}
    assert all(result[0] == tmp_path for result in prepared)
    assert all(result[2] == items for result in prepared)


async def test_hybrid_discovery_lock_serializes_concurrent_rounds(monkeypatch):
    evaluator = AttestedReplayDevEvaluatorV2(
        epoch_id=1,
        worker_index=0,
        snapshot_uri="/unused",
    )
    active = 0
    max_active = 0
    invocations = 0

    async def discover(**kwargs):
        nonlocal active, max_active, invocations
        assert kwargs["cohort_hash"] == "sha256:" + "3" * 64
        invocations += 1
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return {}, (), 0, 0

    monkeypatch.setattr(evaluator, "_discover_provider_overlay_locked", discover)
    assert evaluator._discovery_lock is None

    await asyncio.gather(
        evaluator._discover_provider_overlay(
            candidates=(),
            cohort_hash="sha256:" + "3" * 64,
            remaining_tree_budget_microusd=100_000,
        ),
        evaluator._discover_provider_overlay(
            candidates=(),
            cohort_hash="sha256:" + "3" * 64,
            remaining_tree_budget_microusd=100_000,
        ),
    )

    assert invocations == 2
    assert max_active == 1


async def test_runtime_replay_miss_escalates_and_reruns_the_full_cohort(
    monkeypatch,
):
    tree_id = "sha256:" + "1" * 64

    def candidate(character):
        artifact = SimpleNamespace(
            image_digest=IMAGE_DIGEST,
            model_artifact_hash="sha256:" + character * 64,
            manifest_hash="sha256:" + character.upper().lower() * 64,
        )
        return SimpleNamespace(
            tree_id=tree_id,
            node_id="tree-node:" + character * 64,
            iteration=1,
            draft=SimpleNamespace(
                unified_diff=(
                    "--- a/sourcing_model/logic.py\n"
                    "+++ b/sourcing_model/logic.py\n"
                    "@@ -1 +1 @@\n"
                    "-LOOKUP_VALUE = compose_value('old')\n"
                    "+LOOKUP_VALUE = compose_value('new')\n"
                ),
                target_files=("sourcing_model/logic.py",),
            ),
            build=SimpleNamespace(
                candidate_model_manifest=artifact,
                source_diff_hash="sha256:" + "7" * 64,
            ),
        )

    candidates = (candidate("a"), candidate("b"))
    evaluator = AttestedReplayDevEvaluatorV2(
        epoch_id=24000,
        worker_index=0,
        snapshot_uri="/unused",
        evaluation_concurrency=2,
    )
    calls = []
    initial_roots = {
        candidates[0].node_id: "sha256:" + "2" * 64,
        candidates[1].node_id: "sha256:" + "3" * 64,
    }
    final_roots = {
        candidates[0].node_id: "sha256:" + "4" * 64,
        candidates[1].node_id: "sha256:" + "5" * 64,
    }
    overlay_graph = {
        "root_receipt_hash": "sha256:" + "6" * 64,
        "receipts": [],
    }
    frozen_caches = {
        "cache-ref": {
            "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
            "entries": {"request": {"status": 200, "body_text": "{}"}},
        }
    }

    async def evaluate_candidate(candidate, **kwargs):
        calls.append(
            {
                "node_id": candidate.node_id,
                **kwargs,
            }
        )
        mode = kwargs["evaluation_mode"]
        misses = 1 if mode == "replay" and candidate is candidates[0] else 0
        root = (
            initial_roots[candidate.node_id]
            if mode == "replay"
            else final_roots[candidate.node_id]
        )
        return {
            "result": {
                "snapshot_miss_count": misses,
                "true_miss_count": misses,
                "evaluation_mode": mode,
                "cohort_hash": kwargs["cohort_hash"],
                "overlay_hash": (
                    kwargs["overlay_hash"] if mode == "hybrid" else sha256_json({})
                ),
            },
            "receipt_graph": {
                "root_receipt_hash": root,
                "receipts": [],
            },
        }

    discovery = {}

    async def discover_provider_overlay(**kwargs):
        discovery.update(kwargs)
        return frozen_caches, (overlay_graph,), 3, 12_345

    monkeypatch.setattr(evaluator, "_evaluate_candidate", evaluate_candidate)
    monkeypatch.setattr(
        evaluator,
        "_discover_provider_overlay",
        discover_provider_overlay,
    )

    result = await evaluator.evaluate_cohort(
        candidates,
        remaining_tree_budget_microusd=100_000,
    )

    replay_calls = [call for call in calls if call["evaluation_mode"] == "replay"]
    hybrid_calls = [call for call in calls if call["evaluation_mode"] == "hybrid"]
    assert len(replay_calls) == 2
    assert len(hybrid_calls) == 2
    assert discovery["candidates"] == [candidates[0]]
    assert discovery["remaining_tree_budget_microusd"] == 100_000
    assert result["evaluation_mode"] == "hybrid"
    assert result["provider_call_count"] == 3
    assert result["settled_cost_microusd"] == 12_345
    assert result["overlay_hash"] == sha256_json(frozen_caches)
    assert {call["cohort_hash"] for call in hybrid_calls} == {
        result["cohort_hash"]
    }
    assert result["cohort_hash"] != replay_calls[0]["cohort_hash"]
    expected_parent_roots = {
        *initial_roots.values(),
        overlay_graph["root_receipt_hash"],
    }
    assert all(
        {
            graph["root_receipt_hash"] for graph in call["parent_graphs"]
        }
        == expected_parent_roots
        for call in hybrid_calls
    )
    rows = {row["node_id"]: row for row in result["results"]}
    escalated_plan = rows[candidates[0].node_id]["evaluation_metadata"][
        "evaluation_plan"
    ]
    sibling_plan = rows[candidates[1].node_id]["evaluation_metadata"][
        "evaluation_plan"
    ]
    assert escalated_plan["mode"] == "hybrid"
    assert "strict_replay_miss_requires_live_overlay" in escalated_plan[
        "reason_codes"
    ]
    assert sibling_plan["mode"] == "replay"


async def test_hybrid_discovery_restores_tree_caps_and_limits_live_icps(
    monkeypatch,
):
    calls = []

    class ReplayStore:
        miss_policy = "strict"

        @staticmethod
        def load_manifest():
            return {"manifest_hash": "sha256:" + "9" * 64}

    class FakeRunner:
        def __init__(self, **kwargs):
            self.cache_ref = ""
            self.generated = {}

        async def run_with_provider_evidence(self, icp, context, **kwargs):
            self.cache_ref = icp_evidence_cache_key(icp)
            self.generated = {
                "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
                "icp_ref": self.cache_ref,
                "entries": {},
            }
            calls.append(dict(kwargs))
            return []

        def generated_provider_evidence_cache(self, cache_ref):
            assert cache_ref == self.cache_ref
            return dict(self.generated)

        @staticmethod
        def attested_authorities():
            return [{"receipt_graph": {"root_receipt_hash": "sha256:" + "8" * 64}}]

        @staticmethod
        def provider_evidence_summary(cache_ref):
            return {
                "cost_summary": {
                    "paid_call_count": 0,
                    "total_cost_usd": 0.0,
                    "tracking_failed_count": 0,
                    "cap_blocked": False,
                    "cap_exceeded_after_success": False,
                }
            }

    evaluator = AttestedReplayDevEvaluatorV2(
        epoch_id=1,
        worker_index=0,
        snapshot_uri="/unused",
        live_provider_call_cap=32,
        live_cost_cap_microusd=500_000,
        live_max_icps_per_node=3,
        prior_provider_call_count=10,
        prior_settled_cost_microusd=100_000,
        model_runner_factory=FakeRunner,
    )
    evaluator._dev_items = _dev_items(8)

    async def prepared():
        return (
            Path("/unused"),
            ReplayStore(),
            {
                "source_tree_hash": "sha256:" + "7" * 64,
                "archive_sha256": "sha256:" + "6" * 64,
            },
        )

    monkeypatch.setattr(evaluator, "_ensure_prepared", prepared)
    candidate = SimpleNamespace(
        tree_id="sha256:" + "5" * 64,
        node_id="tree-node:" + "4" * 64,
        build=SimpleNamespace(
            candidate_model_manifest=SimpleNamespace(image_digest=IMAGE_DIGEST)
        ),
    )
    caches, _graphs, paid_calls, cost = (
        await evaluator._discover_provider_overlay_locked(
            candidates=(candidate,),
            cohort_hash="sha256:" + "3" * 64,
            remaining_tree_budget_microusd=250_000,
        )
    )

    assert len(caches) == 3
    assert len(calls) == 3
    assert paid_calls == 0
    assert cost == 0
    assert {call["provider_call_cap"] for call in calls} == {22}
    assert {call["provider_cost_cap_microusd"] for call in calls} == {250_000}


async def test_hybrid_discovery_reuses_identical_sibling_request_once(monkeypatch):
    invocations = []

    class ReplayStore:
        miss_policy = "strict"

        @staticmethod
        def load_manifest():
            return {"manifest_hash": "sha256:" + "9" * 64}

    class FakeRunner:
        def __init__(self, **kwargs):
            self.cache_ref = ""
            self.generated = {}
            self.cost_summary = {}

        async def run_with_provider_evidence(self, icp, context, **kwargs):
            self.cache_ref = icp_evidence_cache_key(icp)
            incoming = dict(kwargs["provider_evidence_cache"])
            if incoming:
                assert incoming["entries"]["shared-request"]["status"] == 200
                paid_calls = 0
                cost_usd = 0.0
                self.generated = incoming
            else:
                paid_calls = 1
                cost_usd = 0.01
                self.generated = {
                    "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
                    "icp_ref": self.cache_ref,
                    "entries": {
                        "shared-request": {
                            "status": 200,
                            "body_text": '{"results":[]}',
                        }
                    },
                }
            self.cost_summary = {
                "paid_call_count": paid_calls,
                "total_cost_usd": cost_usd,
                "tracking_failed_count": 0,
                "cap_blocked": False,
                "cap_exceeded_after_success": False,
            }
            invocations.append(
                {
                    "node_id": context["run_label"],
                    "incoming_cache": incoming,
                    "provider_call_cap": kwargs["provider_call_cap"],
                    "provider_cost_cap_microusd": kwargs[
                        "provider_cost_cap_microusd"
                    ],
                }
            )
            return []

        def generated_provider_evidence_cache(self, cache_ref):
            assert cache_ref == self.cache_ref
            return dict(self.generated)

        @staticmethod
        def attested_authorities():
            return [
                {"receipt_graph": {"root_receipt_hash": "sha256:" + "8" * 64}}
            ]

        def provider_evidence_summary(self, cache_ref):
            assert cache_ref == self.cache_ref
            return {"cost_summary": dict(self.cost_summary)}

    evaluator = AttestedReplayDevEvaluatorV2(
        epoch_id=1,
        worker_index=0,
        snapshot_uri="/unused",
        live_provider_call_cap=32,
        live_cost_cap_microusd=500_000,
        live_max_icps_per_node=1,
        model_runner_factory=FakeRunner,
    )
    evaluator._dev_items = _dev_items(8)

    async def prepared():
        return (
            Path("/unused"),
            ReplayStore(),
            {
                "source_tree_hash": "sha256:" + "7" * 64,
                "archive_sha256": "sha256:" + "6" * 64,
            },
        )

    monkeypatch.setattr(evaluator, "_ensure_prepared", prepared)
    tree_id = "sha256:" + "5" * 64

    def candidate(character):
        return SimpleNamespace(
            tree_id=tree_id,
            node_id="tree-node:" + character * 64,
            build=SimpleNamespace(
                candidate_model_manifest=SimpleNamespace(
                    image_digest=IMAGE_DIGEST
                )
            ),
        )

    caches, _graphs, paid_calls, cost = (
        await evaluator._discover_provider_overlay_locked(
            candidates=(candidate("1"), candidate("2")),
            cohort_hash="sha256:" + "3" * 64,
            remaining_tree_budget_microusd=250_000,
        )
    )

    assert len(caches) == 1
    assert len(invocations) == 2
    assert invocations[0]["incoming_cache"] == {}
    assert invocations[1]["incoming_cache"] == caches[next(iter(caches))]
    assert paid_calls == 1
    assert cost == 10_000
    assert evaluator._tree_paid_call_count == 1
    assert evaluator._tree_cost_microusd == 10_000
    assert {call["provider_call_cap"] for call in invocations} == {32}
    assert {
        call["provider_cost_cap_microusd"] for call in invocations
    } == {250_000}


def test_attested_evaluator_rejects_prior_usage_above_cap():
    with pytest.raises(DevEvalRunnerError, match="prior tree provider-call"):
        AttestedReplayDevEvaluatorV2(
            epoch_id=1,
            worker_index=0,
            snapshot_uri="/unused",
            live_provider_call_cap=2,
            prior_provider_call_count=3,
        )


async def test_evaluator_books_docker_failures_per_icp_without_aborting(tmp_path):
    items = _dev_items(8)
    root = _write_snapshot_set(tmp_path, items)

    def _crashing(*, image_digest, icp, context, snapshot_dir, timeout_seconds):
        raise DevEvalRunnerError("adapter exploded")

    evaluator = DockerReplayDevEvaluator(snapshot_uri=root, run_icp_in_docker=_crashing)
    result = await evaluator(_candidate())
    assert result["aggregate_dev_score"] == 0.0
    assert result["failure_count"] == 8
    assert result["icp_count"] == 8
    assert result["eligible"] is False


# ---------------------------------------------------------------------------
# Default docker invocation (replay containers run with networking disabled)
# ---------------------------------------------------------------------------


def test_default_docker_runner_disables_network_and_mounts_read_only(tmp_path, monkeypatch):
    import subprocess

    from gateway.research_lab import dev_eval_runner as runner_module

    captured: dict = {}

    def _fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)
    evaluator = DockerReplayDevEvaluator(snapshot_uri=str(tmp_path))
    result = evaluator._run_icp_in_docker_default(
        image_digest=IMAGE_DIGEST,
        icp=_dev_icp(0),
        context={"dev_eval": True},
        snapshot_dir=tmp_path,
        timeout_seconds=30,
    )
    assert result == []
    command = captured["command"]
    assert IMAGE_DIGEST in command
    # The replay container must have no network: every provider call is
    # served from the mounted snapshot set, so an unpatched HTTP path fails
    # loudly instead of reaching live providers.
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    mounts = [command[i + 1] for i, part in enumerate(command) if part == "-v"]
    assert any(mount.endswith(":ro") for mount in mounts)
    payload = json.loads(captured["input"])
    assert payload["icp"]["icp_id"] == "dev-0"
