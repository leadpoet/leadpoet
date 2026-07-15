"""Tests for the code_loop_engine fixes from fableanalysis.md.

Covers: bug #5 (checkpoint rehydration + resume restore), bug #16/#17/#20/#21
kill-switch flags, §6.2-5 within-run memory sanitization, and node-id
uniqueness (the S3 diff-key overwrite half of bug #20).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import types

import pytest

from gateway.research_lab import code_loop_engine as engine
from gateway.research_lab.code_build import (
    CodeEditBuildResult,
    CodeEditPatchApplyError,
    ParentImageSourceContext,
)
from gateway.research_lab.git_tree_models import TreePolicy, derive_tree_id
from gateway.research_lab.git_tree_repository import GitTreeCommit
from gateway.research_lab.git_tree_scheduler import (
    GitTreeScheduler,
    GitTreeSchedulerError,
)
from gateway.research_lab.source_symbol_index import build_source_symbol_index
from gateway.research_lab.autoresearch_runtime import (
    AutoResearchRuntimeSettings,
    OpenRouterCallResult,
)
from leadpoet_canonical.attested_v2 import sha256_json
from research_lab.code_editing import CodeEditDraft
from research_lab.eval import PrivateModelArtifactManifest


_PRODUCTION_PLAN_CASES = json.loads(
    (
        Path(__file__).parent
        / "fixtures"
        / "research_lab"
        / "autoresearch_plan_binding_cases.json"
    ).read_text(encoding="utf-8")
)["cases"]

_TREE_TEST_SNAPSHOT_HASH = "sha256:" + "4" * 64
_TREE_TEST_DEV_SET_HASH = "sha256:" + "5" * 64
_TREE_TEST_RECEIPT_ROOT = "sha256:" + "6" * 64


def _draft(**overrides):
    payload = dict(
        failure_mode="weak sourcing recall",
        mechanism="widen provider query fan-out",
        expected_improvement="+2 companies per ICP",
        risk="slower sourcing",
        lane="provider",
        target_files=("sourcing_model/discovery.py",),
        unified_diff=(
            "diff --git a/sourcing_model/discovery.py b/sourcing_model/discovery.py\n"
            "--- a/sourcing_model/discovery.py\n"
            "+++ b/sourcing_model/discovery.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        ),
        redacted_summary="widen fan-out",
        test_plan="run adapter smoke",
        rollback_plan="revert diff",
        predicted_delta=1.5,
        plan_path_id="path-1",
        plan_alignment={"aligned": True},
    )
    payload.update(overrides)
    return CodeEditDraft(**payload)


def _manifest():
    return PrivateModelArtifactManifest(
        model_artifact_hash="sha256:" + "a" * 64,
        git_commit_sha="b" * 40,
        image_digest="sha256:" + "c" * 64,
        config_hash="sha256:" + "d" * 64,
        component_registry_version="1.0",
        scoring_adapter_version="1.0",
        manifest_uri="s3://test-bucket/research-lab/sourcing-model/manifest.json",
        manifest_hash="sha256:" + "e" * 64,
        signature_ref="kms://sig",
        build_id="build-1",
    )


def _tree_lineage(*, node_id: str, source_diff_hash: str) -> dict:
    return {
        "schema_version": "research_lab.git_tree_lineage.v1",
        "tree_id": "sha256:" + "2" * 64,
        "node_id": node_id,
        "parent_node_id": "root",
        "root_branch_id": node_id,
        "depth": 1,
        "child_slot": 0,
        "git_commit": "3" * 64,
        "branch_objective_path_id": "path-1",
        "branch_objective_hash": "sha256:" + "7" * 64,
        "generation_attempt_count": 1,
        "root_artifact_hash": _manifest().model_artifact_hash,
        "parent_artifact_hash": _manifest().model_artifact_hash,
        "parent_dev_score": None,
        "parent_dev_feedback_hash": "",
        "incremental_source_diff_hash": source_diff_hash,
        "cumulative_source_diff_hash": source_diff_hash,
        "composition": {
            "schema_version": "research_lab.git_tree_composition.v1",
            "mode": "direct_parent_git_commit",
            "branch_objective_path_id": "path-1",
            "branch_objective_hash": "sha256:" + "7" * 64,
            "generation_attempt_count": 1,
        },
    }


def _build_result(*, git_tree: dict | None = None):
    source_diff_hash = "sha256:" + "f" * 64
    return CodeEditBuildResult(
        candidate_model_manifest=_manifest(),
        code_edit_manifest={"target_files": ["sourcing_model/discovery.py"], "kind": "code_edit"},
        source_diff_hash=source_diff_hash,
        build_doc={
            "build_doc_hash": "sha256:" + "1" * 64,
            "source_diff_artifact_uri": "s3://test-bucket/diff.json",
            **({"git_tree": dict(git_tree)} if git_tree else {}),
        },
    )


def _source_context(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    (source_root / "sourcing_model").mkdir(exist_ok=True)
    (source_root / "sourcing_model" / "discovery.py").write_text(
        "x = 1\n\n"
        "def discover_companies(query):\n"
        "    \"\"\"Find companies through the current discovery route.\"\"\"\n"
        "    return query\n",
        encoding="utf-8",
    )
    planner_source_index = build_source_symbol_index(
        source_root=source_root,
        editable_files=("sourcing_model/discovery.py",),
        source_tree_hash="sha256:" + "8" * 64,
        parent_image_digest_hash="sha256:" + "9" * 64,
    )
    return ParentImageSourceContext(
        source_root=source_root,
        source_mode="extracted",
        parent_image_digest_hash="sha256:" + "9" * 64,
        source_tree_hash="sha256:" + "8" * 64,
        top_level_paths=("sourcing_model/",),
        editable_files=("sourcing_model/discovery.py",),
        file_previews=(),
        planner_source_index=planner_source_index,
    )


def _production_fixture_source_context(tmp_path, plan):
    base_context = _source_context(tmp_path)
    source_root = base_context.source_root
    selected = next(
        path
        for path in plan["ranked_paths"]
        if path["path_id"] == plan["selected_path_id"]
    )
    symbols_by_path = {}
    for reference in selected["must_inspect"]:
        if "::" in reference:
            path, symbol = reference.split("::", 1)
        elif ".py:" in reference:
            path, symbol = reference.split(":", 1)
        else:
            path, symbol = reference, ""
        symbols_by_path.setdefault(path, set())
        if symbol:
            symbols_by_path[path].add(symbol)
    for relative_path, symbols in symbols_by_path.items():
        target = source_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = ["VALUE = 1", ""]
        for symbol in sorted(symbols):
            assert "." not in symbol
            lines.extend([f"def {symbol}(value):", "    return value", ""])
        target.write_text("\n".join(lines), encoding="utf-8")
    editable_files = tuple(
        sorted({*base_context.editable_files, *symbols_by_path.keys()})
    )
    planner_source_index = build_source_symbol_index(
        source_root=source_root,
        editable_files=editable_files,
        source_tree_hash=base_context.source_tree_hash,
        parent_image_digest_hash=base_context.parent_image_digest_hash,
    )
    return ParentImageSourceContext(
        source_root=source_root,
        source_mode=base_context.source_mode,
        parent_image_digest_hash=base_context.parent_image_digest_hash,
        source_tree_hash=base_context.source_tree_hash,
        top_level_paths=("model/", "sourcing_model/"),
        editable_files=editable_files,
        file_previews=(),
        planner_source_index=planner_source_index,
    )


class _FakeS3:
    def __init__(self, store):
        self.store = store

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.store[(Bucket, Key)] = Body

    def get_object(self, *, Bucket, Key):
        body = self.store[(Bucket, Key)]
        return {"Body": types.SimpleNamespace(read=lambda: body)}


@pytest.fixture()
def fake_boto3(monkeypatch):
    store: dict = {}
    fake = types.ModuleType("boto3")
    fake.client = lambda name: _FakeS3(store)
    monkeypatch.setitem(sys.modules, "boto3", fake)
    return store


class _MemoryTreeArtifactIO:
    def __init__(self):
        self.documents = {}

    def write_json(self, *, uri, document, content_hash, artifact_kind):
        assert uri.startswith("s3://")
        assert artifact_kind == "loop_candidate_rehydration"
        self.documents[uri] = json.loads(json.dumps(document))
        return {"uri": uri, "content_hash": content_hash, "persisted": True}

    def read_json(self, *, uri, content_hash):
        document = json.loads(json.dumps(self.documents[uri]))
        assert document["loop_candidate_artifact_hash"] == content_hash
        return document


class _MemoryTreeRepository:
    def __init__(self):
        self.tree_id = ""
        self.root_commit = "7" * 64
        self.operations = {}
        self.nodes = {}
        self.commits = {}
        self.checkpoints = {}
        self.selection = None
        self.failure = None

    def initialize(self, **kwargs):
        tree_doc = dict(kwargs["tree_doc"])
        self.tree_id = derive_tree_id(
            run_id=kwargs["run_id"],
            root_artifact_hash=kwargs["root_artifact_hash"],
            policy=TreePolicy.from_mapping(tree_doc["policy"]),
        )
        return self.root_commit

    def plan_slot(self, *, slot, request_hash, operation_id, node_doc):
        existing = self.operations.get(operation_id)
        if existing is not None:
            return {
                "created": False,
                "operation_status": existing["status"],
            }
        self.operations[operation_id] = {
            "tree_id": slot.tree_id,
            "node_id": slot.node_id,
            "operation_kind": "generation",
            "slot": slot.to_dict(),
            "request_hash": request_hash,
            "operation_id": operation_id,
            "node_doc": dict(node_doc),
            "status": "reserved",
        }
        return {"created": True, "operation_status": "reserved"}

    def reserve_operation(
        self,
        *,
        operation_id,
        operation_kind,
        request_hash,
        node_id="",
        reservation_doc=None,
    ):
        existing = self.operations.get(operation_id)
        if existing is not None:
            return {
                "created": False,
                "operation_status": existing["status"],
            }
        self.operations[operation_id] = {
            "tree_id": self.tree_id,
            "node_id": node_id,
            "operation_kind": operation_kind,
            "request_hash": request_hash,
            "operation_id": operation_id,
            "reservation_doc": dict(reservation_doc or {}),
            "status": "reserved",
        }
        return {"created": True, "operation_status": "reserved"}

    def inspect_operation(self, *, operation_id):
        operation = self.operations.get(operation_id)
        if operation is None:
            return {"exists": False, "operation_id": operation_id}
        return {
            "exists": True,
            "operation_id": operation_id,
            "operation": json.loads(json.dumps(operation)),
        }

    def operation_settlement_commitment(self):
        terminal = [
            operation
            for operation in self.operations.values()
            if operation["status"] != "reserved"
        ]
        assert len(terminal) == len(self.operations)
        document = {
            "schema_version": "research_lab.git_tree_operation_settlements.v1",
            "tree_id": self.tree_id,
            "operations": [
                {
                    "operation_id": operation["operation_id"],
                    "node_id": operation.get("node_id", ""),
                    "operation_kind": operation["operation_kind"],
                    "status": operation["status"],
                    "request_hash": operation["request_hash"],
                    "result_hash": operation.get("result_hash", ""),
                    "settled_cost_microusd": operation.get(
                        "settled_cost_microusd", 0
                    ),
                    "provider_call_count": operation.get(
                        "provider_call_count", 0
                    ),
                }
                for operation in sorted(
                    terminal, key=lambda item: item["operation_id"]
                )
            ],
        }
        return {
            "operation_count": len(terminal),
            "settled_cost_microusd": sum(
                item.get("settled_cost_microusd", 0) for item in terminal
            ),
            "provider_call_count": sum(
                item.get("provider_call_count", 0) for item in terminal
            ),
            "operation_settlement_hash": sha256_json(document),
        }

    def settle_operation(
        self,
        *,
        operation_id,
        operation_status,
        request_hash,
        result_hash,
        settled_cost_microusd,
        provider_call_count,
        settlement_doc,
    ):
        operation = self.operations[operation_id]
        assert operation["request_hash"] == request_hash
        if operation["status"] != "reserved":
            assert operation["status"] == operation_status
            return {"created": False, "record": dict(operation)}
        operation.update(
            {
                "status": operation_status,
                "result_hash": result_hash,
                "settled_cost_microusd": settled_cost_microusd,
                "provider_call_count": provider_call_count,
                "settlement_doc": json.loads(json.dumps(settlement_doc)),
            }
        )
        return {"created": True, "record": dict(operation)}

    def commit_child(self, *, slot, draft, expected_parent_source_tree_hash):
        assert expected_parent_source_tree_hash
        patch = draft.unified_diff.rstrip() + "\n"
        patch_hash = sha256_json({"unified_diff": patch})
        parent_git_commit = (
            self.root_commit
            if slot.parent_node_id == "root"
            else self.nodes[slot.parent_node_id]["git_commit"]
        )
        commit = GitTreeCommit(
            tree_id=slot.tree_id,
            node_id=slot.node_id,
            parent_node_id=slot.parent_node_id,
            root_branch_id=slot.root_branch_id,
            depth=slot.depth,
            slot_index=slot.slot_index,
            git_commit=sha256_json(
                {"tree_id": slot.tree_id, "node_id": slot.node_id}
            ).split(":", 1)[1],
            parent_git_commit=parent_git_commit,
            source_tree_hash=_manifest().model_artifact_hash,
            draft_patch_hash=sha256_json(
                {"unified_diff": draft.unified_diff}
            ),
            incremental_patch=patch,
            incremental_patch_hash=patch_hash,
            cumulative_patch=patch,
            cumulative_patch_hash=patch_hash,
            changed_files=tuple(draft.target_files),
        )
        self.commits[slot.node_id] = commit
        return commit

    def verify_node_identity(self, **kwargs):
        return {"verified": True, **kwargs}

    def record_node(self, *, node_doc):
        self.nodes[node_doc["node_id"]] = json.loads(json.dumps(node_doc))
        return {"created": True}

    def commit_checkpoint(self, *, checkpoint_hash, checkpoint_doc):
        self.checkpoints[checkpoint_hash] = json.loads(json.dumps(checkpoint_doc))
        return {"created": True}

    def publish_bundle(self):
        return {
            "bundle_uri": f"s3://test-bucket/trees/{self.tree_id}/tree.bundle",
            "bundle_hash": "sha256:" + "8" * 64,
            "bundle_size_bytes": 1024,
            "readback_verified": True,
        }

    def select_final(self, *, selection_hash, selection_doc):
        assert sha256_json(dict(selection_doc)) == selection_hash
        self.selection = json.loads(json.dumps(selection_doc))
        return {"created": True}

    def fail_tree(self, *, failure_hash, failure_doc):
        assert sha256_json(dict(failure_doc)) == failure_hash
        assert self.selection is None
        self.failure = json.loads(json.dumps(failure_doc))
        return {"created": True}


class _CrashAfterBuildRepository(_MemoryTreeRepository):
    def __init__(self):
        super().__init__()
        self.crash_once = True

    def settle_operation(self, **kwargs):
        operation = self.operations[kwargs["operation_id"]]
        if (
            self.crash_once
            and operation["operation_kind"] == "generation"
            and kwargs["operation_status"] == "succeeded"
            and any(
                item["operation_kind"] == "build"
                and item["status"] == "succeeded"
                for item in self.operations.values()
            )
        ):
            self.crash_once = False
            raise RuntimeError("injected crash after committed build")
        return super().settle_operation(**kwargs)


class _MemoryTreeEvaluator:
    async def evaluate_cohort(
        self, candidates, *, remaining_tree_budget_microusd=None
    ):
        assert remaining_tree_budget_microusd is not None
        cohort_hash = sha256_json(
            {"node_ids": sorted(candidate.node_id for candidate in candidates)}
        )
        rows = []
        for candidate in candidates:
            score = 70.0 + int(candidate.node_id[-2:], 16) / 1000.0
            overlay_hash = sha256_json({})
            commitment = {
                "schema_version": "research_lab.git_tree_dev_score_commitment.v1",
                "dev_score_version": "tree-test-v1",
                "dev_set_hash": _TREE_TEST_DEV_SET_HASH,
                "snapshot_manifest_hash": _TREE_TEST_SNAPSHOT_HASH,
                "miss_policy": "strict",
                "evaluation_mode": "replay",
                "overlay_hash": overlay_hash,
                "cohort_hash": cohort_hash,
            }
            rows.append(
                {
                    "node_id": candidate.node_id,
                    "result": {
                        "aggregate_dev_score": score,
                        "dev_score_version": "tree-test-v1",
                        "snapshot_manifest_hash": _TREE_TEST_SNAPSHOT_HASH,
                        "dev_set_hash": _TREE_TEST_DEV_SET_HASH,
                        "eligible": True,
                        "icp_count": 8,
                        "execution_coverage": 1.0,
                        "scored_icp_count": 8,
                        "snapshot_miss_count": 0,
                        "true_miss_count": 0,
                        "failure_count": 0,
                        "zero_output_count": 0,
                        "miss_policy": "strict",
                        "evaluation_mode": "replay",
                        "overlay_hash": overlay_hash,
                        "cohort_hash": cohort_hash,
                        "provider_call_count": 0,
                        "settled_cost_microusd": 0,
                        "score_commitment": sha256_json(commitment),
                        "per_icp": [
                            {
                                "dev_score": score,
                                "company_count": 2,
                                "scored_company_count": 2,
                            }
                            for _ in range(8)
                        ],
                    },
                    "receipt_graph": {
                        "root_receipt_hash": _TREE_TEST_RECEIPT_ROOT
                    },
                }
            )
        return {"results": rows}


def _tree_runtime_policy(policy: TreePolicy) -> dict:
    return {
        "schema_version": "research_lab.git_tree_runtime_policy.v1",
        "policy": policy.to_dict(),
        "evaluator_enabled": True,
        "evaluator_commitment": {
            "snapshot_manifest_hash": _TREE_TEST_SNAPSHOT_HASH,
            "dev_set_hash": _TREE_TEST_DEV_SET_HASH,
            "evaluation_timeout_seconds": 60,
        },
    }


def _tree_test_runtime_policy():
    return _tree_runtime_policy(TreePolicy(
        mode="active",
        branch_factor=1,
        beam_width=1,
        max_depth=1,
        max_nodes=1,
        shortlist_size=1,
        diversity_floor=1,
        deadline_seconds=300,
        finalization_reserve_seconds=30,
    ))


def _model_prompt_context(messages) -> dict:
    payload = messages[-1]["content"].split("Context JSON:\n", 1)[1]
    payload = payload.split("\n\nBounded fallback pass:", 1)[0]
    return json.loads(payload)


@pytest.fixture(autouse=True)
def _tree_evaluator_environment(monkeypatch):
    monkeypatch.setenv(
        "RESEARCH_LAB_DEV_SNAPSHOT_URI",
        "s3://test-bucket/dev-snapshots/READY.json",
    )
    monkeypatch.setenv("RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED", "true")
    original_run = engine.CodeEditLoopEngine.run

    async def run_with_measured_tree(instance, **kwargs):
        instance.settings = replace(
            instance.settings,
            max_seconds=max(120, instance.settings.max_seconds),
        )
        instance.artifact_io = getattr(
            instance, "_test_tree_artifact_io", None
        ) or _MemoryTreeArtifactIO()
        instance.tree_repository = getattr(
            instance, "_test_tree_repository", None
        ) or _MemoryTreeRepository()
        instance.dev_evaluator = getattr(
            instance, "_test_tree_evaluator", None
        ) or _MemoryTreeEvaluator()
        budget_context = dict(kwargs["budget_context"])
        budget_context.setdefault("tree_policy", _tree_test_runtime_policy())
        kwargs["budget_context"] = budget_context
        return await original_run(instance, **kwargs)

    monkeypatch.setattr(
        engine.CodeEditLoopEngine,
        "run",
        run_with_measured_tree,
    )


# --- bug #5: rehydration artifact write + restore round-trip ---


def test_write_and_rehydrate_candidate_round_trip(fake_boto3):
    node_id = "tree-node:" + "4" * 64
    lineage = _tree_lineage(
        node_id=node_id,
        source_diff_hash="sha256:" + "f" * 64,
    )
    doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(),
            run_id="run-1",
            node_id=node_id,
            iteration=2,
            draft=_draft(),
            build=_build_result(git_tree=lineage),
            git_tree=lineage,
        )
    )
    assert doc["loop_candidate_artifact_uri"].startswith("s3://test-bucket/")
    assert doc["loop_candidate_artifact_hash"]

    (_bucket, _key), body = next(iter(fake_boto3.items()))
    payload = json.loads(body.decode("utf-8"))
    candidate = engine._rehydrated_candidate_from_artifact_payload(payload)
    assert candidate.node_id == node_id
    assert candidate.iteration == 2
    assert candidate.draft == _draft()
    assert candidate.build.source_diff_hash == "sha256:" + "f" * 64
    assert candidate.build.candidate_model_manifest == _manifest()


def test_rehydrate_rejects_hash_mismatch(fake_boto3):
    node_id = "tree-node:" + "4" * 64
    lineage = _tree_lineage(
        node_id=node_id,
        source_diff_hash="sha256:" + "f" * 64,
    )
    asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(),
            run_id="run-1",
            node_id=node_id,
            iteration=1,
            draft=_draft(),
            build=_build_result(git_tree=lineage),
            git_tree=lineage,
        )
    )
    (_ref, body) = next(iter(fake_boto3.items()))
    payload = json.loads(body.decode("utf-8"))
    payload["source_diff_hash"] = "sha256:tampered"
    with pytest.raises(ValueError):
        engine._rehydrated_candidate_from_artifact_payload(payload)


def test_write_skips_non_s3_manifest():
    manifest = PrivateModelArtifactManifest.from_mapping(
        {**_manifest().to_dict(), "manifest_uri": "file:///local/manifest.json"}
    )
    doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=manifest,
            run_id="run-1",
            node_id="node",
            iteration=0,
            draft=_draft(),
            build=_build_result(),
        )
    )
    assert doc == {"loop_candidate_artifact_skipped": "manifest_uri_not_s3"}


def _engine_instance(*, expected_candidate=None):
    instance = engine.CodeEditLoopEngine(
        settings=object(),
        call_openrouter=None,
        event_sink=None,
        builder=None,
    )
    if expected_candidate is not None:
        policy = TreePolicy(
            mode="active",
            branch_factor=1,
            beam_width=1,
            max_depth=1,
            max_nodes=1,
            shortlist_size=1,
            diversity_floor=1,
        )
        scheduler = GitTreeScheduler(
            tree_id=expected_candidate.tree_id,
            policy=policy,
        )
        scheduler.record_node(
            engine._candidate_tree_node(expected_candidate, status="evaluating")
        )
        instance._active_tree_policy = policy
        instance._active_tree_id = expected_candidate.tree_id
        instance._active_tree_scheduler = scheduler
    return instance


def test_tree_cohort_deadline_preserves_explicit_post_evaluation_reserve():
    node_id = "tree-node:" + "4" * 64
    policy = TreePolicy(
        mode="active",
        branch_factor=1,
        beam_width=1,
        max_depth=1,
        max_nodes=1,
        shortlist_size=1,
        diversity_floor=1,
        deadline_seconds=300,
        finalization_reserve_seconds=30,
    )
    candidate = engine.BuiltCodeEditCandidate(
        draft=_draft(),
        build=_build_result(),
        node_id=node_id,
        iteration=1,
    )

    class RecordingEvaluator(_MemoryTreeEvaluator):
        def __init__(self):
            self.calls = 0

        async def evaluate_cohort(self, candidates, **kwargs):
            self.calls += 1
            return await super().evaluate_cohort(candidates, **kwargs)

    evaluator = RecordingEvaluator()
    instance = _engine_instance()
    instance._tree_policy_doc = _tree_runtime_policy(policy)
    instance.dev_evaluator = evaluator

    refused = asyncio.run(
        instance._maybe_dev_eval_cohort(
            [candidate],
            run_id="run-deadline-refused",
            remaining_seconds=90,
            remaining_tree_budget_microusd=1_000_000,
            post_evaluation_reserve_seconds=90,
        )
    )
    assert evaluator.calls == 0
    assert refused[0].dev_score is None
    assert refused[0].dev_evaluation["eligibility_reason"] == (
        "insufficient_deadline_for_cohort_evaluation"
    )
    assert refused[0].dev_evaluation["post_evaluation_reserve_seconds"] == 90

    evaluated = asyncio.run(
        instance._maybe_dev_eval_cohort(
            [candidate],
            run_id="run-deadline-allowed",
            remaining_seconds=91,
            remaining_tree_budget_microusd=1_000_000,
            post_evaluation_reserve_seconds=90,
        )
    )
    assert evaluator.calls == 1
    assert evaluated[0].dev_score is not None
    assert evaluated[0].dev_evaluation["eligible"] is True


def test_tree_cohort_settlement_is_allocated_once_and_preserves_generation_cost():
    first = engine.BuiltCodeEditCandidate(
        draft=_draft(),
        build=_build_result(),
        node_id="tree-node:" + "1" * 64,
        iteration=1,
        dev_evaluation={
            "settled_cost_microusd": 5,
            "provider_call_count": 1,
        },
        tree_settled_cost_microusd=10,
    )
    second = replace(
        first,
        node_id="tree-node:" + "2" * 64,
        tree_settled_cost_microusd=20,
    )

    allocated, settled_cost, provider_calls = (
        engine._apply_tree_cohort_settlement([second, first])
    )

    assert [item.node_id for item in allocated] == [first.node_id, second.node_id]
    assert [item.tree_settled_cost_microusd for item in allocated] == [13, 22]
    assert sum(
        item.tree_settled_cost_microusd - prior
        for item, prior in zip(allocated, (10, 20))
    ) == 5
    assert settled_cost == 5
    assert provider_calls == 1

    with pytest.raises(
        GitTreeSchedulerError, match="inconsistent shared accounting"
    ):
        engine._apply_tree_cohort_settlement(
            [
                first,
                replace(
                    second,
                    dev_evaluation={
                        "settled_cost_microusd": 6,
                        "provider_call_count": 1,
                    },
                ),
            ]
        )


def test_tree_engine_forwards_bounded_build_timeout(monkeypatch):
    captured = {}

    class Builder:
        def build(self, *, timeout_seconds, **_kwargs):
            captured["timeout_seconds"] = timeout_seconds
            return "built"

    monkeypatch.setenv("RESEARCH_LAB_LOOP_BUILD_HEARTBEAT", "false")
    instance = engine.CodeEditLoopEngine(
        settings=object(),
        call_openrouter=None,
        event_sink=None,
        builder=Builder(),
    )
    result = asyncio.run(
        instance._build_candidate_inner(
            draft=_draft(),
            artifact=_manifest(),
            run_id="run-build-timeout",
            candidate_index=0,
            source_context=None,
            node_id="tree-node:" + "4" * 64,
            iteration=1,
            elapsed=lambda: 0.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
            build_timeout_seconds=17,
        )
    )
    assert result == "built"
    assert captured == {"timeout_seconds": 17}


def test_run_uses_module_asyncio_without_accidental_local_shadowing():
    assert "asyncio" not in engine.CodeEditLoopEngine.run.__code__.co_varnames


def test_restore_selected_from_resume(fake_boto3):
    node_id = "tree-node:" + "4" * 64
    lineage = _tree_lineage(
        node_id=node_id,
        source_diff_hash="sha256:" + "f" * 64,
    )
    write_doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(),
            run_id="run-1",
            node_id=node_id,
            iteration=3,
            draft=_draft(),
            build=_build_result(git_tree=lineage),
            git_tree=lineage,
        )
    )
    (_ref, body) = next(iter(fake_boto3.items()))
    expected_candidate = engine._rehydrated_candidate_from_artifact_payload(
        json.loads(body.decode("utf-8"))
    )
    resume = {
        "selected_candidates": [
            {
                "node_id": node_id,
                "rehydration_artifact_uri": write_doc["loop_candidate_artifact_uri"],
                "rehydration_artifact_hash": write_doc["loop_candidate_artifact_hash"],
            },
            # Legacy summary without rehydration refs degrades silently.
            {"node_id": "node-legacy"},
        ]
    }
    restored = asyncio.run(
        _engine_instance(expected_candidate=expected_candidate)._restore_selected_from_resume(
            resume=resume,
            run_id="run-1",
            artifact=_manifest(),
            elapsed=lambda: 0.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
        )
    )
    assert len(restored) == 1
    assert restored[0].node_id == node_id
    assert restored[0].iteration == 3
    assert restored[0].rehydration_artifact_uri == write_doc["loop_candidate_artifact_uri"]


def test_restore_rejects_hash_mismatch_for_committed_tree_node(fake_boto3):
    node_id = "tree-node:" + "4" * 64
    lineage = _tree_lineage(
        node_id=node_id,
        source_diff_hash="sha256:" + "f" * 64,
    )
    write_doc = asyncio.run(
        engine._write_private_loop_candidate_artifact(
            artifact=_manifest(),
            run_id="run-1",
            node_id=node_id,
            iteration=1,
            draft=_draft(),
            build=_build_result(git_tree=lineage),
            git_tree=lineage,
        )
    )
    (_ref, body) = next(iter(fake_boto3.items()))
    expected_candidate = engine._rehydrated_candidate_from_artifact_payload(
        json.loads(body.decode("utf-8"))
    )
    resume = {
        "selected_candidates": [
            {
                "node_id": node_id,
                "rehydration_artifact_uri": write_doc["loop_candidate_artifact_uri"],
                "rehydration_artifact_hash": "sha256:not-the-right-hash",
            }
        ]
    }
    with pytest.raises(GitTreeSchedulerError):
        asyncio.run(
            _engine_instance(
                expected_candidate=expected_candidate
            )._restore_selected_from_resume(
                resume=resume,
                run_id="run-1",
                artifact=_manifest(),
                elapsed=lambda: 0.0,
                openrouter_calls=0,
                estimated_cost=0.0,
                actual_cost_microusd=0,
            )
        )


def test_restore_tolerates_missing_selected_candidates():
    restored = asyncio.run(
        _engine_instance()._restore_selected_from_resume(
            resume={},
            run_id="run-1",
            artifact=_manifest(),
            elapsed=lambda: 0.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
        )
    )
    assert restored == []


# --- kill-switch flags (bugs 5, 16, 17, 20, 21; §6.2-5) ---


@pytest.mark.parametrize(
    "helper,env",
    [
        (engine._resume_restore_selected_enabled, "RESEARCH_LAB_LOOP_RESUME_RESTORE_SELECTED"),
        (engine._planner_parse_retry_enabled, "RESEARCH_LAB_LOOP_PLANNER_PARSE_RETRY"),
        (engine._stage_error_containment_enabled, "RESEARCH_LAB_LOOP_STAGE_ERROR_CONTAINMENT"),
        (engine._judge_parse_soft_skip_enabled, "RESEARCH_LAB_LOOP_JUDGE_PARSE_SOFT_SKIP"),
        (engine._within_run_memory_enabled, "RESEARCH_LAB_LOOP_WITHIN_RUN_MEMORY"),
        (engine._min_runtime_skip_when_selected_enabled, "RESEARCH_LAB_LOOP_MIN_RUNTIME_SKIP_WHEN_SELECTED"),
        (engine._build_heartbeat_enabled, "RESEARCH_LAB_LOOP_BUILD_HEARTBEAT"),
    ],
)
def test_fix_flags_default_on_and_disableable(helper, env, monkeypatch):
    monkeypatch.delenv(env, raising=False)
    assert helper() is True
    monkeypatch.setenv(env, "false")
    assert helper() is False


# --- §6.2-5: within-run memory text sanitization ---


def test_memory_safe_text_truncates_and_passes_clean_text():
    text = engine._memory_safe_text("judge rejected: diff does not compile " + "x" * 500)
    assert len(text) <= 280
    assert text.startswith("judge rejected")


def test_memory_safe_text_redacts_secret_shaped_reasons():
    assert "sk-or-" not in engine._memory_safe_text("failed auth with sk-or-abc123")
    assert engine._memory_safe_text("error mentioning service_role denied") == (
        "[redacted secret-like diagnostic text]"
    )


# --- reimbursement cost evidence: failed/no-candidate exits keep miner spend ---


class _NoCandidateBuilder:
    def __init__(self, source_context):
        self._source_context = source_context
        self.config = types.SimpleNamespace(
            loop_planner_enabled=False,
            loop_planner_max_tokens=2000,
            loop_alignment_judge_enabled=False,
            loop_alignment_judge_max_tokens=800,
            loop_novelty_strict=False,
            code_edit_source_inspection_rounds=1,
            code_edit_source_inspection_max_files=4,
            code_edit_source_inspection_file_bytes=4000,
            code_edit_source_inspection_total_bytes=16000,
            code_edit_source_inspection_search_matches=5,
            code_edit_patch_repair_attempts=0,
            planner_reference_repair_enabled=False,
            planner_reference_repair_max_attempts=1,
            provider_capability_catalog_enabled=False,
        )

    def prepare_parent_source_context(self, *, parent_artifact, workspace_dir):
        return self._source_context

    def validate_draft_against_source_context(self, draft, source_context, *, read_paths, require_read):
        return []

    def check_patch_applies(self, *, draft, parent_artifact, source_context):
        return None

    def build(self, *, draft, parent_artifact, run_id, candidate_index, source_context):
        raise AssertionError("no-candidate test should not build")


class _PlannerNoCandidateBuilder(_NoCandidateBuilder):
    def __init__(self, source_context):
        super().__init__(source_context)
        self.config.loop_planner_enabled = True


class _PlannerBuildsCandidateBuilder(_PlannerNoCandidateBuilder):
    def __init__(self, source_context):
        super().__init__(source_context)
        self.builds = []

    def build(self, *, draft, parent_artifact, run_id, candidate_index, source_context):
        self.builds.append((draft.plan_path_id, candidate_index))
        source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
        return CodeEditBuildResult(
            candidate_model_manifest=_manifest(),
            code_edit_manifest={
                "target_files": list(draft.target_files),
                "kind": "code_edit",
            },
            source_diff_hash=source_diff_hash,
            build_doc={
                "build_doc_hash": "sha256:" + "1" * 64,
                "source_diff_artifact_uri": "s3://test-bucket/diff.json",
            },
        )


@pytest.mark.parametrize(
    "fixture_case",
    _PRODUCTION_PLAN_CASES,
    ids=[item["case"] for item in _PRODUCTION_PLAN_CASES],
)
async def test_sanitized_production_plans_bind_and_reach_patch_drafting(
    tmp_path,
    fixture_case,
):
    plan = fixture_case["plan"]
    source_context = _production_fixture_source_context(tmp_path, plan)
    builder = _PlannerBuildsCandidateBuilder(source_context)
    builder.config.planner_reference_repair_enabled = True
    calls = []
    events = []
    selected = next(
        path
        for path in plan["ranked_paths"]
        if path["path_id"] == plan["selected_path_id"]
    )
    first_reference = selected["must_inspect"][0]
    inspection_path = first_reference.split("::", 1)[0].split(":", 1)[0]

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(plan),
                provider_usage={"provider": "fixture", "response_id": fixture_case["case"]},
                cost_microusd=1,
            )
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "requests": [
                            {"operation": "read_file", "path": inspection_path},
                            {"operation": "read_file", "path": "sourcing_model/discovery.py"},
                        ]
                    }
                ),
                provider_usage={"provider": "fixture", "response_id": "inspection"},
                cost_microusd=1,
            )
        if stage == "code_edit_draft":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "candidates": [
                            _draft(
                                lane=selected["lane"],
                                plan_path_id=selected["path_id"],
                                mechanism="improve intent evidence source routing",
                                redacted_summary="improve intent evidence source quality",
                            ).to_dict()
                        ]
                    }
                ),
                provider_usage={"provider": "fixture", "response_id": "draft"},
                cost_microusd=1,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=1,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=builder,
    ).run(
        run_id=f"run-{fixture_case['case']}",
        ticket={
            "ticket_id": f"ticket-{fixture_case['case']}",
            "miner_hotkey": "fixture-hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "fixture-brief",
            "ticket_doc": {"brief_public_summary": "sanitized fixture objective"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="fixture/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "completed"
    assert "source_inspection" in calls
    assert "code_edit_draft" in calls
    assert "loop_planner_reference_repair" not in calls
    planned = next(event for event in events if event.event_type == "loop_direction_planned")
    normalized = planned.event_doc["loop_direction_plan"]
    normalized_selected = next(
        path
        for path in normalized["ranked_paths"]
        if path["path_id"] == normalized["selected_path_id"]
    )
    assert normalized["required_mechanism"] == normalized_selected["mechanism"]
    assert normalized["must_inspect"] == normalized_selected["must_inspect"]
    assert all("::" in reference for reference in normalized["must_inspect"])
    assert any(event.event_type == "code_edit_drafted" for event in events)


@pytest.mark.parametrize("repeat_index", range(20))
async def test_complete_default_tree_has_direct_children_and_one_finalist(
    tmp_path,
    repeat_index,
):
    source_context = _source_context(tmp_path)
    builder = _PlannerBuildsCandidateBuilder(source_context)
    repository = _MemoryTreeRepository()
    events = []
    calls = []
    branch_contexts = []
    ranked_paths = [
        {
            "path_id": "alternate_discovery_surface",
            "lane": "source_routing",
            "mechanism": "add one bounded alternate discovery route",
            "target_behavior": ["recover completed-empty primary searches"],
            "must_inspect": ["sourcing_model/discovery.py::discover_companies"],
            "allowed_lanes": ["source_routing"],
            "disallowed_lanes": ["provider_fallback"],
            "must_not_try": ["do not weaken ICP constraints"],
            "success_criteria": ["preserve exact company-fit checks"],
            "novelty_requirements": ["use a distinct discovery surface"],
            "anti_overfit_checks": ["work across future ICPs"],
            "validation_mode": "runtime_checks",
            "validation_paths": [],
        },
        {
            "path_id": "bounded_query_variant",
            "lane": "query_construction",
            "mechanism": "add one bounded query formulation",
            "target_behavior": ["recover sparse qualified-company searches"],
            "must_inspect": ["sourcing_model/discovery.py::discover_companies"],
            "allowed_lanes": ["query_construction"],
            "disallowed_lanes": ["provider_fallback"],
            "must_not_try": ["do not broaden company-fit gates"],
            "success_criteria": ["preserve exact company-fit checks"],
            "novelty_requirements": ["use a distinct query construction"],
            "anti_overfit_checks": ["work across future ICPs"],
            "validation_mode": "runtime_checks",
            "validation_paths": [],
        },
    ]
    plan = _loop_direction_plan_payload(
        required_lane=ranked_paths[0]["lane"],
        required_mechanism=ranked_paths[0]["mechanism"],
        selected_path_id=ranked_paths[0]["path_id"],
        ranked_paths=ranked_paths,
    )

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        del timeout_seconds, max_tokens
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(plan),
                provider_usage={"provider": "fixture", "response_id": "planner"},
                cost_microusd=1,
            )
        context = _model_prompt_context(messages)
        branch_plan = context["loop_direction_plan"]
        selected_path_id = branch_plan["selected_path_id"]
        selected_path = branch_plan["ranked_paths"][0]
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "requests": [
                            {
                                "operation": "read_file",
                                "path": "sourcing_model/discovery.py",
                            }
                        ]
                    }
                ),
                provider_usage={"provider": "fixture", "response_id": "inspect"},
                cost_microusd=1,
            )
        if stage == "code_edit_draft":
            branch_contexts.append(
                dict(
                    context["budget_context"]["within_run_memory"][
                        "git_tree_branch"
                    ]
                )
            )
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "candidates": [
                            _draft(
                                lane=selected_path["lane"],
                                plan_path_id=selected_path_id,
                                mechanism=selected_path["mechanism"],
                            ).to_dict()
                        ]
                    }
                ),
                provider_usage={"provider": "fixture", "response_id": "draft"},
                cost_microusd=1,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    policy = TreePolicy(
        mode="active",
        deadline_seconds=600,
        finalization_reserve_seconds=30,
    )
    loop = engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=600,
            min_iterations=1,
            max_iterations=6,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=6,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=builder,
    )
    loop._test_tree_repository = repository
    result = await loop.run(
        run_id=f"run-complete-tree-{repeat_index}",
        ticket={
            "ticket_id": f"ticket-complete-tree-{repeat_index}",
            "miner_hotkey": "fixture-hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "fixture-brief",
            "ticket_doc": {"brief_public_summary": "improve qualified sourcing"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="fixture/model",
        budget_context={
            "requested_compute_budget_usd": 5.0,
            "research_model_tier": "default",
            "tree_policy": _tree_runtime_policy(policy),
        },
        requested_loop_count=1,
    )

    nodes = result.tree_result.nodes
    roots = [node for node in nodes if node.parent_node_id == "root"]
    children = [node for node in nodes if node.parent_node_id != "root"]
    assert result.status == "completed"
    assert len(nodes) == 6
    assert len(roots) == 2
    assert len(children) == 4
    assert {node.depth for node in roots} == {1}
    assert {node.depth for node in children} == {2}
    assert {node.parent_node_id for node in children} == {
        node.node_id for node in roots
    }
    assert all(
        repository.commits[node.node_id].parent_git_commit
        == repository.nodes[node.parent_node_id]["git_commit"]
        for node in children
    )
    assert calls.count("loop_planner") == 1
    assert calls.count("source_inspection") == 6
    assert calls.count("code_edit_draft") == 6
    assert len(builder.builds) == 6
    assert len(branch_contexts) == 6
    assert sum(bool(item["parent_feedback"]) for item in branch_contexts) == 4
    assert sum(item["parent_node_id"] == "root" for item in branch_contexts) == 2
    assert len(result.selected_candidates) == 1
    assert result.tree_result.selected_node_id == result.selected_candidates[0].node_id
    assert repository.selection["paid_finalist_count"] == 1
    assert sum(event.event_type == "candidate_selected" for event in events) == 1


def _loop_direction_plan_payload(**overrides):
    payload = {
        "schema_version": "1.0",
        "miner_focus_interpretation": "improve recall without weakening ICP fit",
        "loop_goal": "route to one safe implementation path",
        "required_lane": "source_routing",
        "required_mechanism": "add an alternate discovery surface after completed-empty primary result",
        "generalization_claim": "helps future sealed ICPs with sparse primary-provider coverage",
        "target_behavior": ["preserve ICP constraints", "recover completed-empty primary results"],
        "must_inspect": ["sourcing_model/discovery.py"],
        "allowed_lanes": ["source_routing"],
        "disallowed_lanes": ["provider_fallback"],
        "must_not_try": ["do not weaken ICP constraints"],
        "success_criteria": ["patch touches an editable runtime file"],
        "novelty_requirements": ["not a duplicate of prior attempts"],
        "anti_overfit_checks": ["preserves multiple qualified company outputs"],
        "ranked_paths": [
            {
                "path_id": "alternate_discovery_surface",
                "lane": "source_routing",
                "mechanism": "alternate discovery after completed-empty result",
            }
        ],
        "selected_path_id": "alternate_discovery_surface",
    }
    payload.update(overrides)
    return payload


async def test_restart_restores_a_settled_build_without_regeneration_or_rebuild(
    tmp_path,
):
    repository = _CrashAfterBuildRepository()
    artifact_io = _MemoryTreeArtifactIO()
    evaluator = _MemoryTreeEvaluator()
    builder = _PlannerBuildsCandidateBuilder(_source_context(tmp_path))
    events = []
    calls = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(_loop_direction_plan_payload()),
                provider_usage={"provider": "openrouter", "response_id": "planner"},
                cost_microusd=1000,
            )
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect"},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "candidates": [
                            _draft(
                                lane="source_routing",
                                plan_path_id="alternate_discovery_surface",
                            ).to_dict()
                        ]
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft"},
                cost_microusd=2000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    def new_loop():
        loop = engine.CodeEditLoopEngine(
            settings=AutoResearchRuntimeSettings(
                min_seconds=0,
                max_seconds=30,
                min_iterations=1,
                max_iterations=1,
                draft_timeout_seconds=10,
                reflection_timeout_seconds=10,
                estimated_iteration_cost_usd=0.01,
                max_candidates=1,
            ),
            call_openrouter=call_model,
            event_sink=event_sink,
            builder=builder,
        )
        loop._test_tree_repository = repository
        loop._test_tree_artifact_io = artifact_io
        loop._test_tree_evaluator = evaluator
        return loop

    kwargs = {
        "run_id": "run-build-restart",
        "ticket": {
            "ticket_id": "ticket-build-restart",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "restart-safe build"},
            "requested_loop_count": 1,
        },
        "artifact": _manifest(),
        "component_registry": {},
        "benchmark_public_summary": {"item_count": 1},
        "model_id": "test/model",
        "budget_context": {
            "requested_compute_budget_usd": 5.0,
            "research_model_tier": "default",
        },
        "requested_loop_count": 1,
    }

    with pytest.raises(RuntimeError, match="injected crash"):
        await new_loop().run(**kwargs)

    build_operations = [
        item
        for item in repository.operations.values()
        if item["operation_kind"] == "build"
    ]
    generation_operations = [
        item
        for item in repository.operations.values()
        if item["operation_kind"] == "generation"
    ]
    assert len(build_operations) == 1
    assert build_operations[0]["status"] == "succeeded"
    assert generation_operations[0]["status"] == "reserved"
    assert len(builder.builds) == 1

    checkpoints = [
        event.event_doc["checkpoint"]
        for event in events
        if event.event_type == "checkpoint_saved"
    ]
    assert checkpoints
    calls_before_resume = tuple(calls)
    result = await new_loop().run(
        **kwargs,
        resume_state=checkpoints[-1],
    )

    assert result.status == "completed"
    assert len(result.selected_candidates) == 1
    assert tuple(calls) == calls_before_resume
    assert len(builder.builds) == 1
    assert build_operations[0]["status"] == "succeeded"
    assert generation_operations[0]["status"] == "succeeded"
    assert result.selected_candidates[0].rehydration_artifact_hash


def _loop_direction_plan_v1_1_payload(
    *,
    validation_mode="runtime_checks",
    validation_paths=None,
    **overrides,
):
    path = {
        "path_id": "query_recall_path",
        "lane": "query_construction",
        "mechanism": "add one bounded query variant",
        "target_behavior": ["recover sparse searches"],
        "must_inspect": ["sourcing_model/discovery.py"],
        "allowed_lanes": ["query_construction"],
        "disallowed_lanes": ["provider_fallback"],
        "must_not_try": ["do not weaken ICP constraints"],
        "success_criteria": ["existing validation passes"],
        "novelty_requirements": ["different from prior attempts"],
        "anti_overfit_checks": ["preserve multiple qualified outputs"],
        "validation_mode": validation_mode,
        "validation_paths": list(validation_paths or []),
    }
    payload = {
        "schema_version": "1.1",
        "miner_focus_interpretation": "improve sparse-query recall",
        "loop_goal": "recover qualified companies",
        "required_lane": path["lane"],
        "required_mechanism": path["mechanism"],
        "selected_path_id": path["path_id"],
        "ranked_paths": [path],
        **{key: value for key, value in path.items() if key not in {"path_id", "lane", "mechanism"}},
    }
    payload.update(overrides)
    return payload


async def test_loop_direction_no_new_safe_path_preserves_stop_reason(tmp_path):
    events = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        assert stage == "loop_planner"
        return OpenRouterCallResult(
            content=json.dumps(
                _loop_direction_plan_payload(
                    no_new_safe_path=True,
                    reason="ticket names a concrete provider path not present in editable_files",
                )
            ),
            provider_usage={"provider": "openrouter", "response_id": "planner", "cost_microusd": 1000},
            cost_microusd=1000,
        )

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=6,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_PlannerNoCandidateBuilder(_source_context(tmp_path)),
    ).run(
        run_id="run-no-new-safe-path",
        ticket={
            "ticket_id": "ticket-no-new-safe-path",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "port a missing provider-specific path"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert result.iterations_completed == 0
    assert result.stop_reason == "loop_direction_no_new_safe_path"
    assert [event.event_type for event in events].count("no_viable_patch") == 1
    terminal = events[-1]
    assert terminal.event_type == "loop_failed"
    assert terminal.event_doc["run_summary"]["stop_reason"] == "loop_direction_no_new_safe_path"


async def test_v1_1_infeasible_test_plan_gets_one_bounded_repair(tmp_path):
    events = []
    calls = []
    builder = _PlannerBuildsCandidateBuilder(_source_context(tmp_path))
    builder.config.planner_reference_repair_enabled = True

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(
                    _loop_direction_plan_v1_1_payload(
                        validation_mode="existing_test_files",
                        validation_paths=["tests/test_missing.py"],
                    )
                ),
                provider_usage={"provider": "openrouter", "response_id": "planner"},
                cost_microusd=1000,
            )
        if stage == "loop_planner_reference_repair":
            context = json.loads(messages[1]["content"].split("Context JSON:\n", 1)[1])
            assert context["candidate_edit_constraints"]["editable_test_path_count"] == 0
            assert context["feasibility_errors"]
            return OpenRouterCallResult(
                content=json.dumps(_loop_direction_plan_v1_1_payload()),
                provider_usage={"provider": "openrouter", "response_id": "repair"},
                cost_microusd=1000,
            )
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect"},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "candidates": [
                            _draft(
                                lane="query_construction",
                                plan_path_id="query_recall_path",
                            ).to_dict()
                        ]
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft"},
                cost_microusd=1000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=2,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=builder,
    ).run(
        run_id="run-test-plan-repair",
        ticket={
            "ticket_id": "ticket-test-plan-repair",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "ticket_doc": {"brief_public_summary": "improve query recall"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0},
        requested_loop_count=1,
    )

    assert result.status == "completed"
    assert calls.count("loop_planner_reference_repair") == 1
    assert calls.index("loop_planner_reference_repair") < calls.index("source_inspection")
    assert builder.builds == [("query_recall_path", 0)]
    repair_checkpoints = [
        event.event_doc["checkpoint"]
        for event in events
        if event.event_type == "checkpoint_saved"
        and event.event_doc.get("checkpoint", {}).get("planner_reference_repair_attempted")
    ]
    assert repair_checkpoints[0]["planner_reference_repair_status"] == "attempted"
    assert repair_checkpoints[1]["planner_reference_repair_status"] == "repaired"


async def test_ambiguous_symbol_fails_without_paid_reference_repair(tmp_path):
    source_context = _source_context(tmp_path)
    helper = source_context.source_root / "sourcing_model" / "helper.py"
    helper.write_text(
        "def discover_companies(query):\n    return query\n",
        encoding="utf-8",
    )
    editable_files = (*source_context.editable_files, "sourcing_model/helper.py")
    source_context = ParentImageSourceContext(
        source_root=source_context.source_root,
        source_mode=source_context.source_mode,
        parent_image_digest_hash=source_context.parent_image_digest_hash,
        source_tree_hash=source_context.source_tree_hash,
        top_level_paths=source_context.top_level_paths,
        editable_files=editable_files,
        file_previews=(),
        planner_source_index=build_source_symbol_index(
            source_root=source_context.source_root,
            editable_files=editable_files,
            source_tree_hash=source_context.source_tree_hash,
            parent_image_digest_hash=source_context.parent_image_digest_hash,
        ),
    )
    builder = _PlannerNoCandidateBuilder(source_context)
    builder.config.planner_reference_repair_enabled = True
    plan = _loop_direction_plan_v1_1_payload()
    plan["must_inspect"] = ["discover_companies"]
    plan["ranked_paths"][0]["must_inspect"] = ["discover_companies"]
    calls = []
    events = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(plan),
                provider_usage={"provider": "fixture", "response_id": "ambiguous"},
                cost_microusd=1,
            )
        raise AssertionError(f"unexpected paid stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=1,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=builder,
    ).run(
        run_id="run-ambiguous-reference",
        ticket={
            "ticket_id": "ticket-ambiguous-reference",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "ticket_doc": {"brief_public_summary": "improve discovery"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert calls == ["loop_planner"]
    assert not [
        event
        for event in events
        if event.event_doc.get("stage") == "planner_reference_repair"
    ]
    assert any(
        "loop_direction_plan_reference_ambiguous" in str(event.event_doc.get("error") or "")
        for event in events
    )


async def test_planner_reference_repair_resolves_symbol_and_builds_once(tmp_path):
    events = []
    calls = []
    builder = _PlannerBuildsCandidateBuilder(_source_context(tmp_path))
    builder.config.planner_reference_repair_enabled = True

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(
                    _loop_direction_plan_payload(
                        no_new_safe_path=True,
                        reason="discover_companiez is not present in editable source",
                        unresolved_references=["discover_companiez"],
                    )
                ),
                provider_usage={"provider": "openrouter", "response_id": "planner", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "loop_planner_reference_repair":
            prompt = json.loads(messages[1]["content"].split("Context JSON:\n", 1)[1])
            resolution = prompt["reference_resolution"]
            assert resolution["reference_count"] == 1
            assert resolution["resolved_reference_count"] == 1
            assert resolution["results"][0]["matches"][0]["symbol"] == "discover_companies"
            return OpenRouterCallResult(
                content=json.dumps(
                    _loop_direction_plan_payload(
                        no_new_safe_path=False,
                        reason="",
                        unresolved_references=[],
                        required_mechanism="extend discover_companies without weakening filters",
                    )
                ),
                provider_usage={"provider": "openrouter", "response_id": "repair", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "candidates": [
                            _draft(
                                lane="source_routing",
                                plan_path_id="alternate_discovery_surface",
                            ).to_dict()
                        ]
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft", "cost_microusd": 2000},
                cost_microusd=2000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=3,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=builder,
    ).run(
        run_id="run-reference-repair",
        ticket={
            "ticket_id": "ticket-reference-repair",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "repair one misspelled source reference"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "completed", (
        result.stop_reason,
        calls,
        [
            (
                event.event_type,
                event.event_doc.get("stage"),
                event.event_doc.get("error"),
                event.event_doc.get("reason"),
            )
            for event in events
        ],
    )
    assert calls.count("loop_planner_reference_repair") == 1
    assert len(result.selected_candidates) == 1
    repair_checkpoints = [
        event.event_doc["checkpoint"]
        for event in events
        if event.event_type == "checkpoint_saved"
        and event.event_doc.get("checkpoint", {}).get("planner_reference_repair_attempted")
    ]
    assert [item["stage"] for item in repair_checkpoints[:2]] == [
        "before_planner_reference_repair",
        "after_planner_reference_repair",
    ]
    assert repair_checkpoints[0]["planner_reference_repair_status"] == "attempted"
    assert repair_checkpoints[1]["planner_reference_repair_status"] == "repaired"
    repair_events = [
        event
        for event in events
        if event.event_doc.get("stage") == "planner_reference_repair"
    ]
    serialized = json.dumps([event.event_doc for event in repair_events], sort_keys=True)
    assert "reference_resolution" not in serialized
    assert "Find companies through the current discovery route" not in serialized

    resumed_calls = []
    resumed_events = []
    resumed_builder = _PlannerNoCandidateBuilder(_source_context(tmp_path))
    resumed_builder.config.planner_reference_repair_enabled = True

    async def resumed_call_model(messages, timeout_seconds, max_tokens, stage):
        resumed_calls.append(stage)
        assert stage != "loop_planner_reference_repair"
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect-resume", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage in {"code_edit_draft", "code_edit_fallback"}:
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "no call_sonar implementation is present in editable source",
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": stage, "cost_microusd": 1000},
                cost_microusd=1000,
            )
        raise AssertionError(f"unexpected resumed stage: {stage}")

    async def resumed_event_sink(event):
        resumed_events.append(event)

    resumed_result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=1,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=resumed_call_model,
        event_sink=resumed_event_sink,
        builder=resumed_builder,
    ).run(
            run_id="run-reference-repair",
        ticket={
            "ticket_id": "ticket-reference-repair-resume",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "resume after interrupted repair"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
        resume_state=repair_checkpoints[0],
    )

    assert resumed_result.status == "failed"
    assert "loop_planner_reference_repair" not in resumed_calls
    resumed_checkpoints = [
        event.event_doc["checkpoint"]
        for event in resumed_events
        if event.event_type == "checkpoint_saved"
    ]
    assert resumed_checkpoints[-1]["planner_reference_repair_attempted"] is True


async def test_generation_retry_keeps_the_same_committed_branch_objective(tmp_path):
    events = []
    calls = []
    builder = _PlannerBuildsCandidateBuilder(_source_context(tmp_path))
    builder.config.planner_reference_repair_enabled = True

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(
                    _loop_direction_plan_payload(
                        required_mechanism="extend discover_companiez without weakening filters",
                        selected_path_id="misspelled_discovery_symbol",
                        ranked_paths=[
                            {
                                "path_id": "misspelled_discovery_symbol",
                                "lane": "source_routing",
                                "mechanism": "extend discover_companiez",
                            },
                            {
                                "path_id": "ranked_fallback_path",
                                "lane": "source_routing",
                                "mechanism": "use a separate safe routing path",
                            },
                        ],
                    )
                ),
                provider_usage={"provider": "openrouter", "response_id": "planner"},
                cost_microusd=1000,
            )
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect"},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft" and calls.count("code_edit_draft") == 1:
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "failure_class": "binding_plan_unimplementable",
                        "reason": "discover_companiez is not present in editable source",
                        "missing_references": ["discover_companiez"],
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft-miss"},
                cost_microusd=1000,
            )
        if stage == "code_edit_fallback":
            context_text = messages[-1]["content"].split(
                "Context JSON:\n", 1
            )[1].split("\n\nBounded fallback pass:", 1)[0]
            context = json.loads(context_text)
            plan = context["loop_direction_plan"]
            assert plan["selected_path_id"] == "misspelled_discovery_symbol"
            assert [item["path_id"] for item in plan["ranked_paths"]] == [
                "misspelled_discovery_symbol"
            ]
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "candidates": [
                            _draft(
                                lane="source_routing",
                                plan_path_id="misspelled_discovery_symbol",
                            ).to_dict()
                        ]
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "fallback-fixed"},
                cost_microusd=1000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=2,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=builder,
    ).run(
        run_id="run-draft-reference-repair",
        ticket={
            "ticket_id": "ticket-draft-reference-repair",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "repair a draft source reference"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "completed"
    assert len(result.selected_candidates) == 1
    assert "loop_planner_reference_repair" not in calls
    assert calls.count("code_edit_draft") == 1
    assert calls.count("code_edit_fallback") == 1
    selected = result.selected_candidates[0]
    assert selected.tree_branch_objective_path_id == "misspelled_discovery_symbol"
    assert selected.tree_generation_attempt_count == 2


async def test_unimplementable_branch_stops_at_the_generation_attempt_cap(tmp_path):
    events = []
    calls = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(
                    _loop_direction_plan_payload(
                        required_lane="provider_fallback",
                        required_mechanism="salvage malformed Sonar provider responses",
                        allowed_lanes=["provider_fallback"],
                        disallowed_lanes=["query_construction"],
                        ranked_paths=[
                            {
                                "path_id": "sonar_malformed_200_parse_salvage_recall",
                                "lane": "provider_fallback",
                                "mechanism": "add Sonar parse salvage",
                            }
                        ],
                        selected_path_id="sonar_malformed_200_parse_salvage_recall",
                    )
                ),
                provider_usage={"provider": "openrouter", "response_id": "planner", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "required file is not present in editable_files; no call_sonar implementation exists",
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft", "cost_microusd": 2000},
                cost_microusd=2000,
            )
        if stage == "code_edit_fallback":
            serialized = json.dumps(messages)
            assert "sonar_malformed_200_parse_salvage_recall" in serialized
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "the same committed branch remains unimplementable",
                    }
                ),
                provider_usage={
                    "provider": "openrouter",
                    "response_id": "fallback",
                    "cost_microusd": 1000,
                },
                cost_microusd=1000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    repository = _MemoryTreeRepository()
    loop_engine = engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=6,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_PlannerNoCandidateBuilder(_source_context(tmp_path)),
    )
    loop_engine._test_tree_repository = repository
    result = await loop_engine.run(
        run_id="run-unimplementable-binding-plan",
        ticket={
            "ticket_id": "ticket-unimplementable-binding-plan",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "salvage malformed Sonar responses"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert result.iterations_completed == 1
    assert result.stop_reason == "no_eligible_tree_finalist"
    assert calls.count("code_edit_draft") == 1
    assert calls.count("code_edit_fallback") == 1
    no_viable = [event for event in events if event.event_type == "no_viable_patch"][-1]
    assert (
        no_viable.event_doc["stage"]
        == "candidate_generation_fallback_no_viable_patch"
    )
    retries = [
        event
        for event in events
        if event.event_type == "candidate_generation_fallback_requested"
    ]
    assert len(retries) == 1
    assert retries[0].event_doc["generation_attempt"] == 2
    assert retries[0].event_doc["generation_attempt_limit"] == 2
    terminal = events[-1]
    assert terminal.event_type == "loop_failed"
    assert terminal.event_doc["run_summary"]["stop_reason"] == "no_eligible_tree_finalist"
    assert repository.failure is not None
    assert repository.failure["paid_finalist_count"] == 0
    assert repository.failure["stop_reason"] == "no_eligible_tree_finalist"


async def test_failed_root_branch_does_not_mutate_its_sibling_objective(tmp_path):
    events = []
    calls = []
    objective_calls = []
    builder = _PlannerBuildsCandidateBuilder(_source_context(tmp_path))

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(
                    _loop_direction_plan_payload(
                        required_lane="provider_fallback",
                        required_mechanism="salvage a missing provider-specific function",
                        allowed_lanes=["provider_fallback", "source_routing"],
                        ranked_paths=[
                            {
                                "path_id": "missing_provider_hook",
                                "lane": "provider_fallback",
                                "mechanism": "edit provider hook that is absent",
                            },
                            {
                                "path_id": "source_routing_path",
                                "lane": "source_routing",
                                "mechanism": "widen source routing in an editable file",
                                "target_behavior": ["preserve scoring contract"],
                            },
                        ],
                        selected_path_id="missing_provider_hook",
                    )
                ),
                provider_usage={"provider": "openrouter", "response_id": "planner", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "source_inspection":
            plan = _model_prompt_context(messages)["loop_direction_plan"]
            objective_calls.append(
                (
                    stage,
                    plan["selected_path_id"],
                    tuple(item["path_id"] for item in plan["ranked_paths"]),
                )
            )
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            plan = _model_prompt_context(messages)["loop_direction_plan"]
            objective_calls.append(
                (
                    stage,
                    plan["selected_path_id"],
                    tuple(item["path_id"] for item in plan["ranked_paths"]),
                )
            )
        if (
            stage == "code_edit_draft"
            and plan["selected_path_id"] == "missing_provider_hook"
        ):
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "required file is not present in editable_files; provider hook does not exist",
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft-1", "cost_microusd": 2000},
                cost_microusd=2000,
            )
        if stage == "code_edit_fallback":
            plan = _model_prompt_context(messages)["loop_direction_plan"]
            objective_calls.append(
                (
                    stage,
                    plan["selected_path_id"],
                    tuple(item["path_id"] for item in plan["ranked_paths"]),
                )
            )
            assert plan["selected_path_id"] == "missing_provider_hook"
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "fallback could not repair missing provider hook",
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "fallback", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if (
            stage == "code_edit_draft"
            and plan["selected_path_id"] == "source_routing_path"
        ):
            return OpenRouterCallResult(
                content=json.dumps(
                    {"candidates": [_draft(lane="source_routing", plan_path_id="source_routing_path").to_dict()]}
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft-2", "cost_microusd": 2000},
                cost_microusd=2000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=6,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=builder,
    ).run(
        run_id="run-ranked-path-fallback",
        ticket={
            "ticket_id": "ticket-ranked-path-fallback",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "try alternate ranked paths"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={
            "requested_compute_budget_usd": 5.0,
            "research_model_tier": "default",
            "tree_policy": _tree_runtime_policy(
                TreePolicy(
                    mode="active",
                    branch_factor=2,
                    beam_width=2,
                    max_depth=1,
                    max_nodes=2,
                    shortlist_size=2,
                    diversity_floor=2,
                    deadline_seconds=300,
                    finalization_reserve_seconds=30,
                )
            ),
        },
        requested_loop_count=1,
    )

    assert result.status == "completed"
    assert len(result.selected_candidates) == 1
    assert builder.builds == [("source_routing_path", 0)]
    assert calls.count("code_edit_draft") == 2
    assert calls.count("code_edit_fallback") == 1
    assert objective_calls == [
        (
            "source_inspection",
            "missing_provider_hook",
            ("missing_provider_hook",),
        ),
        (
            "code_edit_draft",
            "missing_provider_hook",
            ("missing_provider_hook",),
        ),
        (
            "code_edit_fallback",
            "missing_provider_hook",
            ("missing_provider_hook",),
        ),
        (
            "source_inspection",
            "source_routing_path",
            ("source_routing_path",),
        ),
        (
            "code_edit_draft",
            "source_routing_path",
            ("source_routing_path",),
        ),
    ]
    assert result.selected_candidates[0].tree_branch_objective_path_id == (
        "source_routing_path"
    )


async def test_single_branch_never_switches_to_an_unplanned_sibling_path(tmp_path):
    events = []
    calls = []
    builder = _PlannerNoCandidateBuilder(_source_context(tmp_path))
    seen_objectives = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(
                    _loop_direction_plan_payload(
                        ranked_paths=[
                            {"path_id": "first_path", "lane": "provider_fallback", "mechanism": "missing hook"},
                            {"path_id": "second_path", "lane": "source_routing", "mechanism": "editable fallback"},
                        ],
                        selected_path_id="first_path",
                    )
                ),
                provider_usage={"provider": "openrouter", "response_id": "planner", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "source_inspection":
            seen_objectives.append(
                _model_prompt_context(messages)["loop_direction_plan"]
            )
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            seen_objectives.append(
                _model_prompt_context(messages)["loop_direction_plan"]
            )
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "required file is not present in editable_files; no hook exists",
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft", "cost_microusd": 2000},
                cost_microusd=2000,
            )
        if stage == "code_edit_fallback":
            seen_objectives.append(
                _model_prompt_context(messages)["loop_direction_plan"]
            )
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "fallback disabled case still has no candidate",
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "fallback", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=6,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=builder,
    ).run(
        run_id="run-ranked-path-disabled",
        ticket={
            "ticket_id": "ticket-ranked-path-disabled",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "fallback disabled"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert result.stop_reason == "no_eligible_tree_finalist"
    assert calls.count("code_edit_draft") == 1
    assert calls.count("code_edit_fallback") == 1
    assert len(seen_objectives) == 3
    assert all(
        plan["selected_path_id"] == "first_path"
        and [item["path_id"] for item in plan["ranked_paths"]]
        == ["first_path"]
        for plan in seen_objectives
    )
    assert result.selected_candidates == ()


async def test_exhausted_generation_attempts_keep_tree_failure_diagnostics(tmp_path):
    events = []
    calls = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(
                    _loop_direction_plan_payload(
                        ranked_paths=[
                            {"path_id": "first_path", "lane": "provider_fallback", "mechanism": "missing hook"},
                            {"path_id": "second_path", "lane": "source_routing", "mechanism": "also missing"},
                        ],
                        selected_path_id="first_path",
                    )
                ),
                provider_usage={"provider": "openrouter", "response_id": "planner", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "required file is not present in editable_files; planned source path missing",
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "draft", "cost_microusd": 2000},
                cost_microusd=2000,
            )
        if stage == "code_edit_fallback":
            return OpenRouterCallResult(
                content=json.dumps(
                    {
                        "no_viable_patch": True,
                        "reason": "fallback could not repair missing ranked path",
                    }
                ),
                provider_usage={"provider": "openrouter", "response_id": "fallback", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=6,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_PlannerNoCandidateBuilder(_source_context(tmp_path)),
    ).run(
        run_id="run-ranked-path-exhausted",
        ticket={
            "ticket_id": "ticket-ranked-path-exhausted",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "all ranked paths fail"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert result.stop_reason == "no_eligible_tree_finalist"
    assert calls.count("code_edit_draft") == 1
    assert calls.count("code_edit_fallback") == 1
    retries = [
        event
        for event in events
        if event.event_type == "candidate_generation_fallback_requested"
    ]
    assert len(retries) == 1
    assert retries[0].event_doc["generation_attempt"] == 2
    assert retries[0].event_doc["generation_attempt_limit"] == 2
    assert events[-1].event_doc["run_summary"]["stop_reason"] == (
        "no_eligible_tree_finalist"
    )


async def test_no_candidate_loop_failed_carries_final_cost_ledger(tmp_path):
    events = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(_loop_direction_plan_payload()),
                provider_usage={"provider": "openrouter", "response_id": "planner"},
                cost_microusd=500,
            )
        if stage == "source_inspection":
            return OpenRouterCallResult(
                content='{"requests":[{"operation":"read_file","path":"sourcing_model/discovery.py"}]}',
                provider_usage={"provider": "openrouter", "response_id": "inspect", "cost_microusd": 1000},
                cost_microusd=1000,
            )
        if stage == "code_edit_draft":
            return OpenRouterCallResult(
                content='{"no_viable_patch":true,"reason":"loop found no safe candidate"}',
                provider_usage={"provider": "openrouter", "response_id": "draft", "cost_microusd": 2000},
                cost_microusd=2000,
            )
        if stage == "code_edit_fallback":
            return OpenRouterCallResult(
                content='{"no_viable_patch":true,"reason":"bounded retry also found no safe candidate"}',
                provider_usage={"provider": "openrouter", "response_id": "fallback"},
                cost_microusd=3000,
            )
        raise AssertionError(f"unexpected stage: {stage}")

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=1,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_PlannerNoCandidateBuilder(_source_context(tmp_path)),
    ).run(
        run_id="run-no-candidate",
        ticket={
            "ticket_id": "ticket-no-candidate",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "brief_sanitized_ref": "brief",
            "ticket_doc": {"brief_public_summary": "test"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={"item_count": 1},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0, "research_model_tier": "default"},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert result.selected_candidates == ()
    assert result.actual_openrouter_cost_microusd == 6500
    terminal = events[-1]
    assert terminal.event_type == "loop_failed"
    assert terminal.cost_ledger["actual_openrouter_cost_microusd"] == 6500
    assert terminal.event_doc["run_summary"]["cost_ledger"]["actual_openrouter_cost_microusd"] == 6500


async def test_final_source_inspection_round_search_only_fails_closed_without_implicit_read(tmp_path):
    events = []
    calls = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        calls.append(stage)
        if stage == "loop_planner":
            return OpenRouterCallResult(
                content=json.dumps(_loop_direction_plan_payload()),
                provider_usage={"provider": "openrouter", "response_id": "planner"},
                cost_microusd=0,
            )
        assert stage == "source_inspection"
        context = json.loads(messages[1]["content"].split("Context JSON:\n", 1)[1])
        assert context["inspection_round"] == 1
        assert context["max_inspection_rounds"] == 1
        assert context["is_final_inspection_round"] is True
        return OpenRouterCallResult(
            content='{"requests":[{"operation":"search","query":"discover_companies"}]}',
            provider_usage={"provider": "openrouter", "response_id": "inspect"},
            cost_microusd=1000,
        )

    async def event_sink(event):
        events.append(event)

    result = await engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=1,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_PlannerNoCandidateBuilder(_source_context(tmp_path)),
    ).run(
        run_id="run-final-search-only",
        ticket={
            "ticket_id": "ticket-final-search-only",
            "miner_hotkey": "hotkey",
            "island": "generalist",
            "ticket_doc": {"brief_public_summary": "inspect query logic"},
            "requested_loop_count": 1,
        },
        artifact=_manifest(),
        component_registry={},
        benchmark_public_summary={},
        model_id="test/model",
        budget_context={"requested_compute_budget_usd": 5.0},
        requested_loop_count=1,
    )

    assert result.status == "failed"
    assert calls == ["loop_planner", "source_inspection"]
    unread = [
        event
        for event in events
        if event.event_doc.get("stage") == "source_inspection_exhausted_without_read"
    ]
    assert len(unread) == 1
    assert unread[0].event_doc["error"] == "code_edit_no_source_files_read"
    assert unread[0].event_doc["inspection_round"] == 1


class _RepairFailureBuilder(_NoCandidateBuilder):
    def __init__(self, source_context):
        super().__init__(source_context)
        self.config.code_edit_patch_repair_attempts = 1

    def check_patch_applies(self, *, draft, parent_artifact, source_context):
        raise CodeEditPatchApplyError("patch does not apply")


class _CostedRepairError(RuntimeError):
    provider_usage = {
        "provider": "openrouter",
        "response_id": "repair-failed",
        "raw_trace_ref": {"s3_ref": "s3://bucket/repair.json", "sha256": "sha256:" + "1" * 64},
    }
    cost_microusd = 4321


async def test_repair_call_failure_records_exception_cost_in_running_ledger(tmp_path):
    events = []

    async def call_model(messages, timeout_seconds, max_tokens, stage):
        assert stage == "code_edit_repair"
        raise _CostedRepairError("repair model failed after provider spend")

    async def event_sink(event):
        events.append(event)

    loop = engine.CodeEditLoopEngine(
        settings=AutoResearchRuntimeSettings(
            min_seconds=0,
            max_seconds=30,
            min_iterations=1,
            max_iterations=1,
            draft_timeout_seconds=10,
            reflection_timeout_seconds=10,
            estimated_iteration_cost_usd=0.01,
            max_candidates=1,
        ),
        call_openrouter=call_model,
        event_sink=event_sink,
        builder=_RepairFailureBuilder(_source_context(tmp_path)),
    )

    repaired, openrouter_calls, estimated_cost, actual_cost_microusd, budget_exhausted = (
        await loop._ensure_patch_applies_or_repair(
            draft=_draft(),
            run_id="run-repair-failed",
            node_id="node-repair-failed",
            iteration=1,
            settings=loop.settings,
            artifact=_manifest(),
            source_context=_source_context(tmp_path),
            source_inspection_context={},
            read_paths=("sourcing_model/discovery.py",),
            budget_context={"requested_compute_budget_usd": 5.0},
            budget_limit_microusd=5_000_000,
            elapsed=lambda: 0.0,
            openrouter_calls=0,
            estimated_cost=0.0,
            actual_cost_microusd=0,
            provider_usage=[],
        )
    )

    assert repaired is None
    assert openrouter_calls == 0
    assert estimated_cost == 0.0
    assert actual_cost_microusd == 4321
    assert budget_exhausted is False
    repair_failure = [
        event
        for event in events
        if event.event_type == "code_edit_repair_failed"
        and event.event_doc.get("stage") == "code_edit_repair_call_failed"
    ][-1]
    assert repair_failure.event_type == "code_edit_repair_failed"
    assert repair_failure.cost_ledger["actual_openrouter_cost_microusd"] == 4321
    assert repair_failure.provider_usage[0]["call_outcome"] == "contained_failure"
    assert repair_failure.provider_usage[0]["raw_trace_ref"]["s3_ref"] == "s3://bucket/repair.json"
    exhausted = [
        event
        for event in events
        if event.event_type == "candidate_repair_exhausted"
        and event.event_doc.get("stage") == "code_edit_repair_call_failed"
    ][-1]
    assert exhausted.cost_ledger["actual_openrouter_cost_microusd"] == 4321
